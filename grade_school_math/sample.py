import argparse
import torch
from dataset import get_examples, GSMDataset, is_correct
from calculator import sample
from transformers import AutoTokenizer, AutoModelForCausalLM


def infer(model, qn, tokenizer, device, sample_len):
    toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
    orig_len = toks["input_ids"].shape[1]
    out = model.generate(
        **toks, max_length=orig_len + sample_len, pad_token_id=tokenizer.eos_token_id,
        do_sample=True, top_k=40, top_p=0.8, temperature=0.7, repetition_penalty=1.05, num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.batch_decode(out)[0]
    return text


# CUDA_VISIBLE_DEVICES=0 python exam.py --model ./model_ckpts_opt1.3/ --tokenizer /nas01/ColossalAI/GLM/models/opt-1.3b

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", "-t", type=str, default="gpt2")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt2",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False, add_bos_token=False,
                                              model_max_length=4096,
                                              padding_side="right", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                 trust_remote_code=True)

    device = torch.device("cuda:0")
    # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    total = len(test_examples)
    corrected = 0
    sample_len = 100
    for item in test_examples:
        text = infer(model, item["question"], tokenizer, device, sample_len)
        print(item["question"])
        if is_correct(text, item):
            corrected += 1

    print(f"model:{args.model}, total:{total}, corrected:{corrected}")
    # print(test_examples.strip())
    # qn = test_examples[1]["question"]
    # print(qn.strip())
    # an = sample(model, qn, tokenizer, device, sample_len)
    # print(f"an:{an}")
    # generator = pipeline('text-generation', model='gpt2')
    # an = generator(qn, max_length=100, num_return_sequences=1, eos_token_id=model.config.eos_token_id)
    # print(f"an2:{an[0]}")
    # isCorrect = is_correct(an[0]["generated_text"], test_examples[1])

    # toks = tokenizer([qn], padding=False, return_tensors="pt").to(device)
    # orig_len = toks["input_ids"].shape[1]
    #
    # out = model.generate(
    #     **toks, max_length=orig_len + 100, pad_token_id=model.config.eos_token_id
    # )
    # text = tokenizer.batch_decode(out)[0]
    # print(text)
    # isCorrect = is_correct(text, test_examples[1])
    # if isCorrect:
    #     print("correct")
    # else:
    #     print("fail")


if __name__ == "__main__":
    main()
