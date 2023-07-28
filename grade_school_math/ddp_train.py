import argparse
import os

import torch as th
from dataset import get_examples, GSMDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
import torch.multiprocessing as mp


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --nnodes=1 ./ddp_train.py --tokenizer /nas01/ColossalAI/GLM/models/opt-1.3b --ckpt_save model_ckpts_opt1.3_ddp --model /nas01/ColossalAI/GLM/models/opt-1.3b

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", "-t", type=str, default="gpt2")
    parser.add_argument("--batch_size", "-bs", type=int, default=8)
    parser.add_argument("--workers", "-w", type=int, default=4)
    parser.add_argument("--ckpt_save", "-save", type=str, default="model_ckpts")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt2",
    )
    args = parser.parse_args()
    return args


def setup(world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    # Initialize the process group
    dist.init_process_group(backend="nccl")


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, add_bos_token=False,
                                              model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=th.bfloat16,
                                                 trust_remote_code=True)

    setup(args.workers)
    local_rank = th.distributed.get_rank()

    th.cuda.set_device(local_rank)
    device = th.device("cuda", local_rank)
    print('using device:{}'.format(device))
    model.to(device)
    model = DDP(model, [local_rank], local_rank)

    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)

    model.train()

    # train_loader = DataLoader(train_dset, batch_size=16, shuffle=True)
    train_sampler = th.utils.data.distributed.DistributedSampler(train_dset)
    train_loader = th.utils.data.DataLoader(train_dset, batch_size=args.batch_size,
                                            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    optim = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 20
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optim,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    pbar = tqdm(range(num_training_steps))
    for epoch in range(num_epochs):
        for batch in train_loader:
            optim.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            loss.backward()
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    if local_rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.ckpt_save)

    cleanup()


if __name__ == "__main__":
    args = parser_args()
    port_id = 29292
    args.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    args.num_gpus = th.cuda.device_count()
    if args.num_gpus == 1:
        main(args=args)
    else:
        th.multiprocessing.set_start_method('spawn')
    mp.spawn(main, nprocs=args.workers, args=(args,))
