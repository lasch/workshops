# Copyright (c) 2022 Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache-style license found in the
# LICENSE file in the root directory of this source tree.
import os

from dataclasses import dataclass
from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch


@dataclass
class train_config:
    # general
    host_port: str = "12368"

    # seed
    seed: int = 2022
    
    
    # model
    # model_name = "google/t5-v1_1-xl"  # << - adjust model size here
    model_name = os.getenv("MODEL_NAME", "3b").lower()
    if model_name == "3b":
        model_name = "google/t5-v1_1-xl"
    elif model_name == "11b":
        model_name = "google/t5-v1_1-xxl"
    else:
        model_name = "google/t5-v1_1-xl"
    
    # available models
    # google/t5-v1_1-small  # 60 M
    # google/t5-v1_1-base   # 223 M
    # google/t5-v1_1-large  # 737 M
    # google/t5-v1_1-xl     # 3 Billion
    # google/t5-v1_1-xxl    # 11 Billion 

    tokenizer = "t5-large"   # no need to adjust, tokenizer works for all model sizes

    # save models
    save_model: bool = False
    checkpoint_max_save_count: int = (
        2  # number of 'best' checkpoints to save based on val loss
    )

    # compile
    use_torch_compile: bool = False

    # sharding policy
    # sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  #FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    sharding_strategy = os.getenv("SHARDING_STRATEGY", "full").lower()
    if sharding_strategy == "full":
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "grad":
        sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "no":
        sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD
    else:
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    print_sharding_plan: bool = False


    # cpu offload
    cpu_offload = os.getenv("CPU_OFFLOAD", "false").lower()
    if cpu_offload == "true":
        cpu_offload = True
    else:
        cpu_offload = False

    # backward prefetch
    # backward_prefetch = BackwardPrefetch.BACKWARD_PRE  #BACKWARD_PRE, BACKWARD_POST
    backward_prefetch = os.getenv("BACKWARD_PREFETCH", "pre").lower()
    if backward_prefetch == "pre":
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif backward_prefetch == "post":
        backward_prefetch = BackwardPrefetch.BACKWARD_POST
    elif backward_prefetch == "none":
        backward_prefetch = None
    else:
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    

    # dataloaders
    num_workers_dataloader: int = 0

    # policies 
    # mixed precision this will default to BFloat16, but if no native support detected, will 
    # use FP16.  (note that FP16 is not recommended for larger models...)
    use_mixed_precision: bool = True

    HF_activation_checkpointing: bool = False
    FSDP_activation_checkpointing: bool = True

    # datasets
    dataset_train = "datasets_grammar/gtrain_150K.csv"
    # dataset_train = "/workspace/data/lchu/gtrain_1M.csv"  # /workspace/data/lchu/gtrain_10M.csv, /workspace/data/lchu/gtrain_150K.csv
    dataset_test = "datasets_grammar/grammar_validation.csv"

    # training
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    num_epochs: int = 2

    # validation
    run_validation: bool = True
    val_batch_size = 8
    block_for_validation: bool = False

    # logging
    track_memory = True
    memory_report: bool = True
    nccl_debug_handler: bool = True
    distributed_debug: bool = True

    # Fine Tuning
    use_child_tuning: bool = False
    learning_rate: float = 4e-8

    use_task_free: bool = True
    use_fisher_matrix: bool = False
    percent_F: float = 0.35
