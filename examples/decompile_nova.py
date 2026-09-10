#!/usr/bin/env python3
"""Decompile named ELF functions with Nova's released BCR checkpoint."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path
from typing import Any, Callable

try:
    from .binary_decompile_common import (
        add_input_arguments,
        extract_functions,
        read_function_names,
        restore_function_name,
        write_results,
    )
except ImportError:  # Support direct script execution.
    from binary_decompile_common import (
        add_input_arguments,
        extract_functions,
        read_function_names,
        restore_function_name,
        write_results,
    )


WORKSPACE = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/home/jiang/model/nova-1.3b-bcr")
AUTHOR_NORMALIZER = WORKSPACE / "third_party/nova/data/normalize.py"


def load_author_normalizer(path: Path = AUTHOR_NORMALIZER) -> Callable[[str], str]:
    spec = importlib.util.spec_from_file_location("nova_author_normalize", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Nova normalizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.normalize


def decompile(
    binary: str | Path,
    function_names: list[str],
    *,
    model_path: str | Path = DEFAULT_MODEL,
    optimization: str = "O3",
    device: str = "cuda",
    max_new_tokens: int = 1024,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    seed: int = 42,
) -> tuple[Path, list[dict[str, Any]]]:
    """Use Nova's normalization, tokenizer, and hierarchical attention mask."""

    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if temperature <= 0 and num_return_sequences != 1:
        raise ValueError("greedy Nova decoding supports one return sequence")
    model_path = Path(model_path).expanduser().resolve()
    sys.path.insert(0, str(model_path))
    try:
        from modeling_nova import NovaForCausalLM, NovaTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "run this example with `.venv-nova/bin/python`; Nova requires "
            "its released Transformers-4.40 architecture"
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    binary_path, functions = extract_functions(Path(binary), function_names)
    normalize = load_author_normalizer()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    nova_tokenizer = NovaTokenizer(tokenizer)
    model = NovaForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    records: list[dict[str, Any]] = []
    for function in functions:
        normalized = normalize(function.raw_objdump).strip()
        if not normalized.startswith("<func0>:"):
            raise ValueError(
                f"Nova normalization failed for {function.name}: "
                "missing <func0> marker"
            )
        assembly = normalized[len("<func0>:") :]
        prompt_before = (
            f"# This is the assembly code with {optimization} optimization:\n"
            "<func0>:"
        )
        prompt_after = "\nWhat is the source code?\n"
        prompt = prompt_before + assembly + prompt_after
        char_types = (
            "0" * len(prompt_before)
            + "1" * len(assembly)
            + "0" * len(prompt_after)
        )
        encoded = nova_tokenizer.encode(prompt, "", char_types)
        input_ids = torch.as_tensor(
            encoded["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        nova_attention_mask = torch.as_tensor(
            encoded["nova_attention_mask"], dtype=torch.long, device=device
        ).unsqueeze(0)
        no_mask_idx = torch.as_tensor(
            [encoded["no_mask_idx"]], dtype=torch.long, device=device
        )
        generation_args: dict[str, Any] = {
            "inputs": input_ids,
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": num_return_sequences,
            "do_sample": temperature > 0,
            "nova_attention_mask": nova_attention_mask,
            "no_mask_idx": no_mask_idx,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_args.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            generated = model.generate(**generation_args)
        candidates = [
            restore_function_name(
                tokenizer.decode(
                    row[input_ids.size(1) :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                ),
                function.name,
            )
            for row in generated
        ]
        records.append(
            {
                "function_name": function.name,
                "source": candidates[0],
                "candidates": candidates,
                "error": None,
                "normalized_assembly": normalized,
                "generation": {"prompt_tokens": int(input_ids.size(1))},
            }
        )
    return binary_path, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--optimization",
        choices=("O0", "O1", "O2", "O3"),
        default="O3",
        help="Must match the optimization used to build the input binary.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = read_function_names(args)
    binary, records = decompile(
        args.binary,
        names,
        model_path=args.model,
        optimization=args.optimization,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_return_sequences=args.num_return_sequences,
        seed=args.seed,
    )
    write_results(
        output=args.output,
        method="Nova",
        binary=binary,
        records=records,
        configuration={
            "model": str(args.model),
            "optimization": args.optimization,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "normalization": str(AUTHOR_NORMALIZER),
        },
    )


if __name__ == "__main__":
    main()
