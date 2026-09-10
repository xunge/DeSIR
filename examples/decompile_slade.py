#!/usr/bin/env python3
"""Decompile named ELF functions with a released x86 SLaDe checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

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
DEFAULT_O0_MODEL = (
    WORKSPACE
    / "third_party/slade_artifact/output/"
    "export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best"
)
DEFAULT_O3_MODEL = (
    WORKSPACE
    / "third_party/slade_artifact/output/"
    "export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best"
)


def encode_assembly(tokenizer, assembly: str, optimization: str) -> list[int]:
    if optimization == "O0":
        text = f"<c> <mask:0> </c> <intel> {assembly} </intel>"
    elif optimization == "O3":
        text = (
            f"<c> <mask:0> </c> <intel> "
            f"<opt3> {assembly} </opt> </intel>"
        )
    else:
        raise ValueError("released SLaDe x86 checkpoints cover only O0 and O3")
    return tokenizer.encode(tokenizer.normalizer.normalize_str(text)).ids


def decode_candidate(tokenizer, token_ids: list[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    text = (
        text.replace("<eol> ", "\n")
        .replace("<eol>", "\n")
        .replace("<tab> ", "\t")
        .replace("<tab>", "\t")
        .replace("<mask:0>", "")
        .replace("<pad>", "")
        .replace("<s>", "")
        .replace("</s>", "")
    )
    text = re.sub(r"# (/\w+)*", "", text)
    return (
        text.replace("0x ", "0x")
        .replace(" #", "")
        .replace("return", "return ")
        .replace("return  ", "return ")
        .replace("static", "")
        .replace("inline", "")
        .replace("__attribute__((used))", "")
        .strip()
    )


def decompile(
    binary: str | Path,
    function_names: list[str],
    *,
    optimization: str = "O3",
    o0_model: str | Path = DEFAULT_O0_MODEL,
    o3_model: str | Path = DEFAULT_O3_MODEL,
    device: str = "cuda:0",
    batch_size: int = 8,
    num_beams: int = 5,
    num_return_sequences: int = 1,
    max_new_tokens: int = 512,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run SLaDe on a reconstructed GAS view of named binary functions."""

    import torch
    from tokenizers import Tokenizer
    from torch.nn.utils.rnn import pad_sequence
    from transformers import BartForConditionalGeneration

    if num_return_sequences > num_beams:
        raise ValueError("num_return_sequences may not exceed num_beams")
    checkpoint = Path(o0_model if optimization == "O0" else o3_model)
    binary_path, functions = extract_functions(Path(binary), function_names)
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = (
        BartForConditionalGeneration.from_pretrained(checkpoint).eval().to(device)
    )
    pad_id = tokenizer.get_vocab()["<pad>"]
    encoded = [
        encode_assembly(tokenizer, function.slade_gas, optimization)
        for function in functions
    ]
    limit = model.config.max_position_embeddings
    eligible = [index for index, ids in enumerate(encoded) if len(ids) <= limit]
    decoded_by_index: dict[int, list[str]] = {}

    for start in range(0, len(eligible), batch_size):
        indices = eligible[start : start + batch_size]
        input_ids = pad_sequence(
            [torch.tensor(encoded[index]) for index in indices],
            batch_first=True,
            padding_value=pad_id,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                early_stopping=True,
                length_penalty=1.0,
                min_length=1,
            )
        generated = generated.view(len(indices), num_return_sequences, -1).cpu()
        for index, hypotheses in zip(indices, generated):
            decoded_by_index[index] = [
                decode_candidate(tokenizer, hypothesis.tolist())
                for hypothesis in hypotheses
            ]

    records: list[dict[str, Any]] = []
    for index, function in enumerate(functions):
        candidates = decoded_by_index.get(index, [])
        restored = [
            restore_function_name(candidate, function.name)
            for candidate in candidates
        ]
        records.append(
            {
                "function_name": function.name,
                "source": restored[0] if restored else "",
                "candidates": restored,
                "error": None if restored else "encoder_context_overflow",
                "assembly": function.slade_gas,
                "generation": {
                    "input_tokens": len(encoded[index]),
                    "encoder_limit": limit,
                },
            }
        )
    return binary_path, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument(
        "--optimization",
        choices=("O0", "O3"),
        default="O3",
        help="Must match the input binary; no released x86 O1/O2 checkpoint.",
    )
    parser.add_argument("--o0-model", type=Path, default=DEFAULT_O0_MODEL)
    parser.add_argument("--o3-model", type=Path, default=DEFAULT_O3_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    names = read_function_names(args)
    binary, records = decompile(
        args.binary,
        names,
        optimization=args.optimization,
        o0_model=args.o0_model,
        o3_model=args.o3_model,
        device=args.device,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        max_new_tokens=args.max_new_tokens,
    )
    checkpoint = args.o0_model if args.optimization == "O0" else args.o3_model
    write_results(
        output=args.output,
        method="SLaDe",
        binary=binary,
        records=records,
        configuration={
            "checkpoint": str(checkpoint),
            "optimization": args.optimization,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "input_adapter": (
                "objdump AT&T instructions with reconstructed local labels "
                "and a SLaDe GAS envelope"
            ),
            "adapter_limitation": (
                "source-level GAS directives do not survive in a binary; "
                "this adapter is outside SLaDe's exact compiler-GAS protocol"
            ),
        },
    )


if __name__ == "__main__":
    main()
