from transformers import Trainer, TrainingArguments
import argparse
import os
from accelerate import Accelerator

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


def collate_fn(examples):
    pixel_values = torch.stack([example[0] for example in examples])
    labels = torch.tensor([example[1] for example in examples])
    return {"x":pixel_values, "labels":labels}


class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs["x"])
        target = inputs["labels"]
        loss = F.nll_loss(outputs, target)
        return (loss, outputs) if return_outputs else loss


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


def setup(rank, world_size):
    "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("zy_sft", backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    "Cleans up the distributed environment"
    dist.destroy_process_group()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, add_bos_token=False,
                                              model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=th.bfloat16, device_map="auto",
                                                 trust_remote_code=True)

    training_args = TrainingArguments(
        "basic-trainer",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        remove_unused_columns=False
    )
    trainer = MyTrainer(
        model,
        training_args,
        train_dataset=train_dset,
        eval_dataset=test_dset,
        data_collator=collate_fn,
    )


if __name__ == "__main__":
    args = parser_args()
    main(args)