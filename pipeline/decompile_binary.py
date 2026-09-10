#!/usr/bin/env python3
"""Run one DeSIR checkpoint on a grounded binary function."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from protocol import extract_c_source, render_prompt, restore_anchors
from ground_binary import build_grounding


DEFAULT_MODELS = {
    "1.3b": "xunge/desir-1.3b",
    "6.7b": "xunge/desir-6.7b",
}


def generate(prompt: str, model_name: str, max_new_tokens: int, temperature: float) -> str:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before running model inference") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    do_sample = temperature > 0
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0, encoded["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


def run(binary: Path, function: str | None, model_name: str, grounding_path: Path | None,
        output: Path, max_new_tokens: int, temperature: float) -> dict:
    if grounding_path:
        record = json.loads(grounding_path.read_text(encoding="utf-8"))
    else:
        record = build_grounding(binary, function, max_strings=32, max_symbols=32)
    prompt = render_prompt(record["assembly"], record["grounding"])
    generated = generate(prompt, model_name, max_new_tokens, temperature)
    source = restore_anchors(extract_c_source(generated), record["grounding"])
    result = {
        "schema": "desir-decompilation-v1",
        "model": model_name,
        "binary": str(binary.resolve()),
        "function": record["function"],
        "prompt": prompt,
        "raw_generation": generated,
        "source": source,
        "output": source,
        "grounding": record["grounding"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--function", default=None)
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--model", default=None, help="Local path or Hugging Face model id")
    parser.add_argument("--size", choices=sorted(DEFAULT_MODELS), default="1.3b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    model = args.model or DEFAULT_MODELS[args.size]
    result = run(args.binary, args.function, model, args.grounding, args.output,
                  args.max_new_tokens, args.temperature)
    print(result["source"])


if __name__ == "__main__":
    main()
