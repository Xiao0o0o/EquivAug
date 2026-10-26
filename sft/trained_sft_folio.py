#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT on FOLIO with full fine-tuning (Qwen2.5-1.5B / Qwen2.5-3B).
For Llama-3.1-8B (LoRA) see trained_sft_folio_lora.py.

Each JSONL line has "premises_nl", "conclusion_nl" and "explanation".
Loss is computed on response tokens only.
"""

import os
import json
import torch
import argparse
import logging
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Alpaca-style prompt with NL premises and conclusion
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Determine whether the conclusion follows from the premises. "
    "Think step by step. Answer True, False, or Uncertain.\n\n"
    "Premises: {premises}\n"
    "Conclusion: {conclusion}\n\n"
    "### Response:\n"
)


class CLMCollator:
    """Data collator for causal language modeling with response-only loss."""
    
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        # Pad input_ids & attention_mask
        input_features = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.tokenizer.pad(
            input_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()

        # Collect raw masks
        raw_masks = [f.get("labels_mask", None) for f in features]

        # If no masks at all -> mask everything (conservative)
        if all(m is None for m in raw_masks):
            labels[:] = -100
            batch["labels"] = labels
            return batch

        # Normalize masks to python lists
        normalized = []
        for m, f in zip(raw_masks, features):
            ids_len = len(f["input_ids"])
            if m is None:
                arr = [0] * ids_len
            else:
                if isinstance(m, list):
                    arr = list(m)
                elif hasattr(m, 'tolist'):
                    arr = m.tolist()
                else:
                    arr = list(m)
                arr = [1 if x else 0 for x in arr]
                if len(arr) < ids_len:
                    arr = arr + [0] * (ids_len - len(arr))
                else:
                    arr = arr[:ids_len]
            normalized.append(arr)

        # Pad masks to batch max length
        max_len = batch["input_ids"].shape[1]
        mask_tensors = []
        for arr in normalized:
            if len(arr) < max_len:
                arr = arr + [0] * (max_len - len(arr))
            else:
                arr = arr[:max_len]
            mask_tensors.append(arr)
        
        labels_mask = torch.tensor(mask_tensors, dtype=torch.bool)

        # Apply mask: only compute loss on response tokens
        labels[~labels_mask] = -100
        
        # Also mask padding tokens
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels

        # Debug output (once)
        if os.environ.get("MASK_DEBUG", "0") == "1" and not hasattr(self, "_printed"):
            self._printed = True
            active_ratio = labels_mask.float().mean(dim=1).tolist()
            print(f"[MASK DEBUG] active_ratio_per_example: {active_ratio}")
            trained_ids = batch["input_ids"][0][labels_mask[0]].tolist()
            print("[MASK DEBUG] decoded_trained_text_example0:")
            print(self.tokenizer.decode(trained_ids, skip_special_tokens=False)[:500])

        return batch


class DataProcessor:
    """Process training data from JSONL format."""
    
    def __init__(self, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def load_data(self, data_path):
        """Load data from JSONL file."""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def format_pair(self, premises, conclusion, answer_text):
        """Format a premises/conclusion pair for base model (no chat template)."""
        prompt_text = PROMPT_TEMPLATE.format(premises=premises, conclusion=conclusion)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos is None:
            raise ValueError("Tokenizer does not have an eos_token defined. Please set it in setup_model_and_tokenizer.")
        
        # Ensure answer ends with eos token
        if not answer_text.endswith(eos):
            answer_text = answer_text + eos

        # Full text (input + response)
        full_text = prompt_text + answer_text
        
        # Debug output (once)
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG] Prompt text sample:\n{prompt_text[:300]}")
            print(f"[DEBUG] Full text sample:\n{full_text[:500]}")
        
        return {"prompt_text": prompt_text, "full_text": full_text}
    
    def create_training_examples(self, data):
        """Create training examples from raw data."""
        examples = []
        
        for item in data:
            premises = item['premises_nl']
            conclusion = item['conclusion_nl']
            answer_text = item['explanation']
            
            example = self.format_pair(premises, conclusion, answer_text)
            examples.append(example)
        
        return examples
    
    def tokenize_function(self, examples):
        """Tokenize examples and create labels mask for response-only loss."""
        # Tokenize full text
        full = self.tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        
        # Tokenize prompt only (to find where response starts)
        prompt_tok = self.tokenizer(
            examples["prompt_text"],
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
        )
        
        # Create labels mask: 0 for prompt, 1 for response
        labels_mask = []
        for ids_full, ids_prompt in zip(full["input_ids"], prompt_tok["input_ids"]):
            prompt_len = len(ids_prompt)
            response_len = len(ids_full) - prompt_len
            mask = [0] * prompt_len + [1] * max(0, response_len)
            mask = mask[:len(ids_full)]
            labels_mask.append(mask)
        
        return {
            "input_ids": full["input_ids"],
            "attention_mask": full["attention_mask"],
            "labels_mask": labels_mask
        }
    
    def prepare_dataset(self, data_path, validation_split=0.1):
        """Prepare dataset with train/validation split."""
        raw = self.load_data(data_path)
        pairs = self.create_training_examples(raw)
        
        logger.info(f"Total training examples: {len(pairs)}")
        
        if validation_split > 0:
            train_pairs, val_pairs = train_test_split(
                pairs, test_size=validation_split, random_state=42
            )
            
            train_ds = Dataset.from_dict({
                "prompt_text": [p["prompt_text"] for p in train_pairs],
                "full_text": [p["full_text"] for p in train_pairs],
            })
            val_ds = Dataset.from_dict({
                "prompt_text": [p["prompt_text"] for p in val_pairs],
                "full_text": [p["full_text"] for p in val_pairs],
            })
            
            train_tok = train_ds.map(
                self.tokenize_function,
                batched=True,
                remove_columns=["prompt_text", "full_text"],
                desc="Tokenizing train"
            )
            val_tok = val_ds.map(
                self.tokenize_function,
                batched=True,
                remove_columns=["prompt_text", "full_text"],
                desc="Tokenizing val"
            )
            
            train_tok.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels_mask"])
            val_tok.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels_mask"])
            
            return train_tok, val_tok
        else:
            ds = Dataset.from_dict({
                "prompt_text": [p["prompt_text"] for p in pairs],
                "full_text": [p["full_text"] for p in pairs],
            })
            tok_ds = ds.map(
                self.tokenize_function,
                batched=True,
                remove_columns=["prompt_text", "full_text"],
                desc="Tokenizing"
            )
            tok_ds.set_format(type="numpy", columns=["input_ids", "attention_mask", "labels_mask"])
            return tok_ds, None


def setup_model_and_tokenizer(model_name, gradient_checkpointing=False):
    """Setup model and tokenizer for full fine-tuning."""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True
    )
    
    # Set pad token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # Enable gradient checkpointing for memory efficiency
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Resize token embeddings if necessary
    model.resize_token_embeddings(len(tokenizer))
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="SFT Training for Base Models")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B",
                        help="Path to pretrained base model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (JSONL file)")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.1,
                        help="Proportion of data for validation")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory for model")
    parser.add_argument("--num_train_epochs", type=float, default=1.0,
                        help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every X steps")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                        choices=["no", "epoch", "steps"],
                        help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save every X steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Max checkpoints to keep")
    parser.add_argument("--eval_strategy", type=str, default="steps",
                        choices=["no", "epoch", "steps"],
                        help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=50,
                        help="Evaluate every X steps")
    parser.add_argument("--fp16", action="store_true",
                        help="Use fp16 training")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing")
    parser.add_argument("--optim", type=str, default="adamw_torch_fused",
                        help="Optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Warmup steps (overrides warmup_ratio if set)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        help="LR scheduler type")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Dataloader workers")
    parser.add_argument("--report_to", type=str, nargs="+", default=["none"],
                        help="Reporting integrations (e.g., 'tensorboard', 'wandb', 'none')")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--debug_mask", action="store_true",
                        help="Enable mask debugging")
    parser.add_argument("--group_by_length", action="store_true",
                        help="Group samples by length (for more efficient training)")
    
    args = parser.parse_args()
    
    # Enable mask debugging if requested
    if args.debug_mask:
        os.environ["MASK_DEBUG"] = "1"
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name_or_path,
        args.gradient_checkpointing
    )
    
    # Prepare dataset
    logger.info(f"Loading data from {args.data_path}")
    processor = DataProcessor(tokenizer, args.max_seq_length)
    train_dataset, eval_dataset = processor.prepare_dataset(
        args.data_path, args.validation_split
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    # Determine precision
    use_fp16 = args.fp16 and not args.bf16
    use_bf16 = args.bf16
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=args.validation_split > 0,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy=args.eval_strategy if args.validation_split > 0 else "no",
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps if args.warmup_steps is not None else 0,
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=args.report_to,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_unused_columns=False,
        group_by_length=args.group_by_length,
    )
    
    # Data collator
    data_collator = CLMCollator(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Training completed!")
    print(f"\nModel saved to: {args.output_dir}")
    print(f"Load with: AutoModelForCausalLM.from_pretrained('{args.output_dir}')")


if __name__ == "__main__":
    main()