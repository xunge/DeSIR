#!/usr/bin/env python3
"""Minimal Transformers Trainer for the prepared DeSIR JSONL format."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


class ExampleDataset:
    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        return {"input_ids": item.input_ids, "labels": item.labels}


def load_examples(path: Path, tokenizer, max_length: int) -> list[Example]:
    examples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + completion_ids + [tokenizer.eos_token_id])[:max_length]
        prompt_len = min(len(prompt_ids), len(ids))
        examples.append(Example(ids, [-100] * prompt_len + ids[prompt_len:]))
    return examples


class Collator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        import torch
        width = max(len(item["input_ids"]) for item in batch)
        ids, masks, labels = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad] * pad)
            masks.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="base model or local checkpoint")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()
    try:
        import torch
        from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                                  TrainingArguments)
    except ImportError as exc:
        raise SystemExit("Install requirements-train.txt before training") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    examples = load_examples(args.train, tokenizer, args.max_length)
    if not examples:
        raise SystemExit("training file contains no usable rows")
    training_args = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, logging_steps=10,
        save_strategy="steps", save_steps=500, bf16=torch.cuda.is_available(),
        report_to=[], remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ExampleDataset(examples),
                      data_collator=Collator(tokenizer))
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved checkpoint to {args.output}")


if __name__ == "__main__":
    main()
