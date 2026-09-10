#!/usr/bin/env python3
"""Run released SLaDe checkpoints on x86 GCC optimization levels.

SLaDe publishes separate BART checkpoints for x86 O0 and x86 O3.  This
runner skips O1/O2 by default.  The explicit ``--use-o3-for-o1-o2`` option
supports a clearly labelled zero-shot generalization experiment when all four
optimization levels are required; it never presents O1/O2 as released models.
The token wrapping, beam search, and detokenization match the CGO'24 artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BartForConditionalGeneration


def gcc_source_assembly(source: str, opt: str, gcc: str) -> str:
    """Recreate the GAS function input used by the released SLaDe artifact."""

    optimization = {
        "O0": "-O0",
        "O1": "-O1",
        "O2": "-O2",
        "O3": "-O3",
    }[opt]
    with tempfile.TemporaryDirectory(prefix="slade-gas-") as tmp:
        source_path = Path(tmp) / "input.c"
        source_path.write_text(source, encoding="utf-8")
        proc = subprocess.run(
            [gcc, "-S", optimization, "-x", "c", "-o", "-", str(source_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-4000:])

    # Match SLaDe's GASCompiler._gas_get_func_asm_from_all_asm: retain the
    # function body through .cfi_endproc and remove '#' comments.
    function = [".globl func0", ".type func0, @function"]
    inside = False
    for line in proc.stdout.splitlines():
        if line.startswith("func0:"):
            inside = True
        if inside:
            without_comment = line.split("#", 1)[0]
            if without_comment.split():
                function.append(without_comment)
        if inside and ".cfi_endproc" in line:
            break
    if not inside or not any(".cfi_endproc" in line for line in function):
        raise RuntimeError("could not extract func0 GAS assembly")
    return "\n".join(function) + "\n"


def encode_assembly(tokenizer: Tokenizer, assembly: str, opt: str) -> list[int]:
    if opt == "O0":
        text = f"<c> <mask:0> </c> <intel> {assembly} </intel>"
    elif opt == "O3":
        text = f"<c> <mask:0> </c> <intel> <opt3> {assembly} </opt> </intel>"
    else:
        raise ValueError(f"SLaDe has no released x86 {opt} checkpoint")
    normalized = tokenizer.normalizer.normalize_str(text)
    return tokenizer.encode(normalized).ids


def decode_candidate(tokenizer: Tokenizer, token_ids: list[int]) -> str:
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
    text = (
        text.replace("0x ", "0x")
        .replace(" #", "")
        .replace("return", "return ")
        .replace("return  ", "return ")
        .replace("static", "")
        .replace("inline", "")
        .replace("__attribute__((used))", "")
    )
    return text.strip()


def save_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def run_level(
    rows: list[dict],
    checkpoint: Path,
    dataset_opt: str,
    checkpoint_opt: str,
    device: str,
    batch_size: int,
    beams: int,
    candidates: int,
    max_new_tokens: int,
    assembly_source: str,
    gcc: str,
) -> list[dict]:
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    model = (
        BartForConditionalGeneration.from_pretrained(checkpoint)
        .eval()
        .to(device)
    )
    pad_id = tokenizer.get_vocab()["<pad>"]
    predictions: list[dict] = []

    # Released SLaDe checkpoints generate C, not C++; keep the language scope
    # explicit when a mixed-language benchmark is supplied.
    level_rows = [
        row
        for row in rows
        if row["type"] == dataset_opt and row.get("language", "c") == "c"
    ]
    for start in tqdm(
        range(0, len(level_rows), batch_size),
        desc=f"SLaDe x86 {dataset_opt} via {checkpoint_opt}",
    ):
        batch_rows = level_rows[start : start + batch_size]
        prepared = []
        for row in batch_rows:
            try:
                assembly = (
                    row["input_asm_prompt"]
                    if assembly_source == "dataset"
                    else gcc_source_assembly(
                        row.get("c_prefix", "") + "\n" + row["c_func"],
                        dataset_opt,
                        gcc,
                    )
                )
                adapted_row = row
                if assembly_source == "gcc-source":
                    adapted_row = {
                        **row,
                        "original_input_asm_prompt": row["input_asm_prompt"],
                        "input_asm_prompt": assembly,
                    }
                prepared.append(
                    (
                        adapted_row,
                        encode_assembly(
                            tokenizer, assembly, checkpoint_opt
                        ),
                    )
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                predictions.append(
                    {
                        **row,
                        "output": "",
                        "candidates": [""] * candidates,
                        "generation": {
                            "method": "slade",
                            "optimization": dataset_opt,
                            "checkpoint_optimization": checkpoint_opt,
                            "zero_shot_optimization_transfer": (
                                dataset_opt != checkpoint_opt
                            ),
                            "checkpoint": str(checkpoint),
                            "assembly_source": assembly_source,
                            "assembly_error": str(exc),
                            "context_overflow": False,
                        },
                    }
                )
        batch_rows = [row for row, _ in prepared]
        encoded = [token_ids for _, token_ids in prepared]
        keep = [
            (row, token_ids)
            for row, token_ids in zip(batch_rows, encoded)
            if len(token_ids) <= model.config.max_position_embeddings
        ]
        too_long = [
            row
            for row, token_ids in zip(batch_rows, encoded)
            if len(token_ids) > model.config.max_position_embeddings
        ]
        for row in too_long:
            predictions.append(
                {
                    **row,
                    "output": "",
                    "candidates": [""] * candidates,
                    "generation": {
                        "method": "slade",
                        "optimization": dataset_opt,
                        "checkpoint_optimization": checkpoint_opt,
                        "zero_shot_optimization_transfer": (
                            dataset_opt != checkpoint_opt
                        ),
                        "checkpoint": str(checkpoint),
                        "assembly_source": assembly_source,
                        "context_overflow": True,
                    },
                }
            )
        if not keep:
            continue

        kept_rows, token_lists = zip(*keep)
        input_ids = pad_sequence(
            [torch.tensor(ids) for ids in token_lists],
            batch_first=True,
            padding_value=pad_id,
        ).to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=beams,
                num_return_sequences=candidates,
                early_stopping=True,
                length_penalty=1.0,
                min_length=1,
            )
        generated = generated.view(len(kept_rows), candidates, -1).cpu()
        for row, hypotheses in zip(kept_rows, generated):
            decoded = [
                decode_candidate(tokenizer, hypothesis.tolist())
                for hypothesis in hypotheses
            ]
            predictions.append(
                {
                    **row,
                    "output": decoded[0],
                    "candidates": decoded,
                    "generation": {
                        "method": "slade",
                        "optimization": dataset_opt,
                        "checkpoint_optimization": checkpoint_opt,
                        "zero_shot_optimization_transfer": (
                            dataset_opt != checkpoint_opt
                        ),
                        "checkpoint": str(checkpoint),
                        "assembly_source": assembly_source,
                        "context_overflow": False,
                        "num_beams": beams,
                        "num_return_sequences": candidates,
                        "max_new_tokens": max_new_tokens,
                    },
                }
            )
    del model
    torch.cuda.empty_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--o0-model", type=Path)
    parser.add_argument("--o3-model", type=Path)
    parser.add_argument(
        "--use-o3-for-o1-o2",
        action="store_true",
        help=(
            "evaluate O1/O2 as explicitly labelled zero-shot transfer "
            "using the released x86-O3 checkpoint"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--assembly-source",
        choices=("dataset", "gcc-source"),
        default="gcc-source",
        help=(
            "gcc-source recompiles c_func to the gcc -S GAS representation "
            "used by the released SLaDe artifact"
        ),
    )
    parser.add_argument("--gcc", default="gcc")
    args = parser.parse_args()
    if not args.o0_model and not args.o3_model:
        parser.error("supply at least one released checkpoint")
    if args.use_o3_for_o1_o2 and not args.o3_model:
        parser.error("--use-o3-for-o1-o2 requires --o3-model")
    if args.num_return_sequences > args.num_beams:
        parser.error("--num-return-sequences cannot exceed --num-beams")

    rows = json.loads(args.dataset.read_text(encoding="utf-8"))
    predictions: list[dict] = []
    levels = [("O0", args.o0_model, "O0")]
    if args.use_o3_for_o1_o2:
        levels.extend(
            [
                ("O1", args.o3_model, "O3"),
                ("O2", args.o3_model, "O3"),
            ]
        )
    levels.append(("O3", args.o3_model, "O3"))
    for dataset_opt, checkpoint, checkpoint_opt in levels:
        if checkpoint:
            predictions.extend(
                run_level(
                    rows,
                    checkpoint,
                    dataset_opt,
                    checkpoint_opt,
                    args.device,
                    args.batch_size,
                    args.num_beams,
                    args.num_return_sequences,
                    args.max_new_tokens,
                    args.assembly_source,
                    args.gcc,
                )
            )
    order = {
        (row.get("task_id", row.get("id")), row["type"]): index
        for index, row in enumerate(rows)
    }
    predictions.sort(
        key=lambda row: order[
            (row.get("task_id", row.get("id")), row["type"])
        ]
    )
    save_atomic(args.output, predictions)
    print(
        f"Wrote {len(predictions)} predictions to {args.output}; "
        + (
            "O1/O2 use explicitly labelled zero-shot transfer from x86-O3"
            if args.use_o3_for_o1_o2
            else "O1/O2 are unsupported by the released SLaDe checkpoints"
        )
    )


if __name__ == "__main__":
    main()
