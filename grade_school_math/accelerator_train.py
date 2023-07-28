import argparse
from accelerate import Accelerator

import torch as th
from dataset import get_examples, GSMDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Config, AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# accelerate config
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch ./accelerator_train.py --tokenizer /nas01/ColossalAI/GLM/models/opt-1.3b --ckpt_save model_ckpts_opt1.3 --model /nas01/ColossalAI/GLM/models/opt-1.3b --batch_size 16
# accelerate launch ./accelerate_train.py --tokenizer /aigc-nas02/hf_models/Llama-2-7b-hf --model /aigc-nas02/hf_models/Llama-2-7b-hf --batch_size 8 --ckpt_save model_ckpts_llama2-7

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
    params = parser.parse_args()
    return params


def main(params):
    tokenizer = AutoTokenizer.from_pretrained(params.tokenizer, use_fast=False, add_bos_token=False,
                                              model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(params.model, torch_dtype=th.bfloat16,
                                                 trust_remote_code=True)
    accelerator = Accelerator(split_batches=True)
    local_rank = th.distributed.get_rank()

    th.cuda.set_device(local_rank)
    device = th.device("cuda", local_rank)
    print('using device:{}'.format(device))


    train_examples = get_examples("train")
    train_dset = GSMDataset(tokenizer, train_examples)
    train_loader = DataLoader(train_dset, shuffle=True, batch_size=params.batch_size)

    # test_examples = get_examples("test")
    # test_dset = GSMDataset(tokenizer, test_examples)
    # test_sampler = th.utils.data.distributed.DistributedSampler(test_dset)
    # test_loader = th.utils.data.DataLoader(train_dset, batch_size=args.batch_size,
    #                                         num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    optim = AdamW(model.parameters(), lr=1e-5)

    train_loader, model, optim = accelerator.prepare(
        train_loader, model, optim
    )
    model.train()

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
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs[0]
            accelerator.backward(loss)
            optim.step()
            lr_scheduler.step()
            pbar.update(1)
            pbar.set_description(f"train_loss: {loss.item():.5f}")

    # net_dict = accelerator.get_state_dict(model)
    # accelerator.save(net_dict, params.ckpt_save + "_" + str(epoch))
    if local_rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(params.ckpt_save)



if __name__ == "__main__":
    params = parser_args()
    port_id = 29292
    params.dist_url = 'tcp://127.0.0.1:' + str(port_id)
    if params.workers == 1:
        main(params=params)
    else:
        mp.set_start_method('spawn')
    mp.spawn(main, nprocs=params.workers, args=params)
