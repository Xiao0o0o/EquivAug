"""
Merge a LoRA adapter (from trained_sft_*_lora.py) into its base model so that
vLLM (evaluation) and verl (GRPO) can load a plain HuggingFace checkpoint.

For Llama-3.x models it also
  - sets eos_token to <|end_of_text|> (what the SFT model learned to emit),
  - writes all three Llama-3.x stop tokens to generation_config.json and config.json,
  - sets the chat template to the SFT prompt format.

Usage:
    python merge_lora.py --adapter_path outputs/sft/llama3.1-8b-gsm8k-lora \
        --output_path outputs/sft/llama3.1-8b-gsm8k-lora-merged
"""

import argparse
import json
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SFT_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{{ message['content'] }}

### Response:
{% elif message['role'] == 'assistant' %}{{ message['content'] }}{% endif %}{% endfor %}"""

# 128001 = <|end_of_text|>, 128008 = <|eom_id|>, 128009 = <|eot_id|>
LLAMA3_EOS_IDS = [128001, 128008, 128009]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    tok = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True, use_fast=True)

    # The SFT scripts resize the embeddings to len(tokenizer); do the same before loading the adapter
    base_name = PeftConfig.from_pretrained(args.adapter_path).base_model_name_or_path
    base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=dtype, device_map="cpu", trust_remote_code=True)
    base.resize_token_embeddings(len(tok))
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_path, safe_serialization=True)
    print(f"[merge] merged adapter into {args.output_path}")

    if "<|end_of_text|>" in tok.get_vocab():
        tok.eos_token = "<|end_of_text|>"
        tok.chat_template = SFT_CHAT_TEMPLATE
        tok.save_pretrained(args.output_path)
        print("[tokenizer] eos_token = <|end_of_text|>, chat_template = SFT template")

        for name in ["generation_config.json", "config.json"]:
            path = os.path.join(args.output_path, name)
            cfg = {}
            if os.path.exists(path):
                with open(path) as f:
                    cfg = json.load(f)
            cfg["eos_token_id"] = LLAMA3_EOS_IDS
            if name == "generation_config.json":
                cfg["pad_token_id"] = 128001
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[{name}] eos_token_id = {LLAMA3_EOS_IDS}")
    else:
        tok.save_pretrained(args.output_path)


if __name__ == "__main__":
    main()
