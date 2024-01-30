from fms.models.hf import to_hf_api
from fms.models.llama import LLaMAConfig, LLaMA
from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    load_state_dict,
)

load_path = "/lustre/t5/workshops/pretraining/ckpt/checkpoints/step_1700000_ckp"
save_path = "/lustre/t5/workshops/pretraining/ckpt/adapted_hf/step_1700000_ckp"

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

model = LLaMA(llama_config, orig_init=True)

state_dict = {"model_state": model.state_dict()}
load_state_dict(
    state_dict=state_dict, storage_reader=FileSystemReader(load_path), no_dist=True
)
model.load_state_dict(state_dict["model_state"])

hf_model = to_hf_api(model)

hf_model.save_pretrained(save_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/lustre/llama_weights/hf/7B")
tokenizer.save_pretrained(save_path)