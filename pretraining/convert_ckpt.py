import math
import os

import fire
import torch
import torch.optim as optim
from fms.models.llama import LLaMAConfig, LLaMA
from fms.utils.checkpointing import Checkpointer
from torch import distributed as dist
from torch.distributed._tensor import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor.parallel.fsdp import enable_2d_with_fsdp
from torch.optim.lr_scheduler import LambdaLR

import config
import policies
from pretraining.utils.config_utils import update_config
from pretraining.utils.dataset_utils import get_data_loader, get_dummy_loader
from pretraining.utils.train_utils import setup, setup_environ_flags, get_policies, train, get_profiler


def main(**kwargs):
    # get configs
    cfg = config.train_config()
    update_config(cfg, **kwargs)

    # ensure reproducibility
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if rank == 0:
        print(f"--> running with these configs {cfg}")

    # some setups
    setup()
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    setup_environ_flags()

    # get policy
    mixed_precision_policy, wrapping_policy, sharding_strategy_policy = get_policies(cfg, rank)

    # get fms model
    llama_config = LLaMAConfig(
        src_vocab_size=32000,
        emb_dim=4096,
        norm_eps=1e-05,
        nheads=32,
        nlayers=32,
        hidden_grow_factor=11008/4096,
        multiple_of=1,
        activation_fn="silu",
        max_expected_seq_len=2048,
    )
    if cfg.low_cpu_fsdp:
        if rank == 0:
            model = LLaMA(llama_config, orig_init=True)
        else:
            with torch.device("meta"):
                model = LLaMA(llama_config, orig_init=True)
    else:
        model = LLaMA(llama_config, orig_init=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> model has {total_params / 1e6} Million params\n")

    # TP
    if cfg.tp_size > 1:
        assert enable_2d_with_fsdp()
        twod_mesh = init_device_mesh("cuda", (world_size // cfg.tp_size, cfg.tp_size))
        blocks = model.get_submodule("layers")
        for i, block in enumerate(blocks):
            if rank == 0:
                print("parallelization of block:", i)
            block = parallelize_module(
                module=block,
                device_mesh=twod_mesh,
                parallelize_plan={
                    "attn.query": ColwiseParallel(),
                    "attn.key": ColwiseParallel(),
                    "attn.value": ColwiseParallel(),
                    "attn.dense": RowwiseParallel(),
                    "ff_sub_layer.w1": ColwiseParallel(),
                    "ff_sub_layer.wg": ColwiseParallel(),
                    "ff_sub_layer.w2": RowwiseParallel(),
                },
                tp_mesh_dim=1,
            )
        fsdp_pg = twod_mesh.get_dim_groups()[0]
    else:
        fsdp_pg = None

    # FSDP
    model = FSDP(
        model,
        process_group=fsdp_pg,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=sharding_strategy_policy,
        use_orig_params=cfg.use_torch_compile,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=cfg.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        if cfg.low_cpu_fsdp and rank != 0 else None,
        device_mesh=init_device_mesh("cuda", (world_size // cfg.sharding_group_size, cfg.sharding_group_size))
        if cfg.sharding_strategy == "hsdp" else None,
    )

    # fsdp activation checkpointing
    if cfg.fsdp_activation_checkpointing:
        if rank == 0:
            print(f"--> applying FSDP activation checkpointing...")
        policies.apply_fsdp_checkpointing(model, cfg.selective_checkpointing)

    if cfg.use_torch_compile:
        print("compile not supported yet for llama ")

    # optionally load from checkpoint
    checkpointer = Checkpointer(cfg.ckpt_save_path, 1000, cfg.sharding_strategy, rank, local_rank)
    model, _, _, step, ntok = checkpointer.load(
        model,
        None,
        None,
        path=cfg.ckpt_load_path
    )

    checkpointer.save_single_file(step, model)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
