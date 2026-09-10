#!/usr/bin/env python3
"""Decompile named ELF functions with FAE + self-constructed-context sc²dec."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from .binary_decompile_common import (
        add_input_arguments,
        direct_prompt,
        extract_c_code,
        extract_function,
        extract_functions,
        read_function_names,
        restore_function_name,
        write_results,
    )
except ImportError:  # Support direct script execution.
    from binary_decompile_common import (
        add_input_arguments,
        direct_prompt,
        extract_c_code,
        extract_function,
        extract_functions,
        read_function_names,
        restore_function_name,
        write_results,
    )


DEFAULT_BASE_MODEL = "/home/jiang/model/llm4decompile-6.7b-v1.5"
DEFAULT_ADAPTER = "/home/jiang/model/sccdec-lora"
C_PREAMBLE = """
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef unsigned long long ull;
"""
C_EXAMPLE = """
bool func0(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) return false;
    }
    return true;
}
""".strip()


def chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if not tokenizer.chat_template:
        raise ValueError(
            "the selected tokenizer has no chat template; use the sc²dec "
            "base checkpoint tokenizer"
        )
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def compile_candidate_to_assembly(source: str, optimization: str) -> str | None:
    """Recompile a func0 prediction for sc²dec's second-pass demonstration."""

    with tempfile.TemporaryDirectory(prefix="sccdec-binary-example-") as tmp:
        tmpdir = Path(tmp)
        source_path = tmpdir / "candidate.c"
        library_path = tmpdir / "candidate.so"
        source_path.write_text(
            C_PREAMBLE + "\n" + extract_c_code(source), encoding="utf-8"
        )
        try:
            proc = subprocess.run(
                [
                    "gcc",
                    "-shared",
                    "-fPIC",
                    f"-{optimization}",
                    str(source_path),
                    "-lm",
                    "-o",
                    str(library_path),
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
            )
            if proc.returncode != 0:
                return None
            return extract_function(library_path, "func0").assembly
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired):
            return None


def decompile(
    binary: str | Path,
    function_names: list[str],
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter: str = DEFAULT_ADAPTER,
    optimization: str = "O3",
    one_shot: bool = True,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 16384,
    max_new_tokens: int = 1024,
    compile_workers: int = 8,
    enforce_eager: bool = False,
    seed: int = 42,
) -> tuple[Path, list[dict[str, Any]]]:
    """Run the released FAE adapter and its self-constructed-context pass."""

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    binary_path, functions = extract_functions(Path(binary), function_names)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    fixed_assembly = (
        compile_candidate_to_assembly(C_EXAMPLE, optimization) if one_shot else None
    )
    if one_shot and fixed_assembly is None:
        raise RuntimeError("failed to compile the fixed sc²dec one-shot example")

    initial_messages: list[list[dict[str, str]]] = []
    for function in functions:
        messages: list[dict[str, str]] = []
        if one_shot:
            messages.extend(
                [
                    {"role": "user", "content": direct_prompt(fixed_assembly)},
                    {"role": "assistant", "content": C_EXAMPLE},
                ]
            )
        messages.append(
            {"role": "user", "content": direct_prompt(function.assembly)}
        )
        initial_messages.append(messages)
    initial_prompts = [
        chat_prompt(tokenizer, messages) for messages in initial_messages
    ]
    token_limit = max_model_len - max_new_tokens
    initial_lengths = [
        len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in initial_prompts
    ]
    initial_eligible = [
        index for index, length in enumerate(initial_lengths) if length <= token_limit
    ]

    llm = LLM(
        model=base_model,
        tokenizer=base_model,
        enable_lora=True,
        max_lora_rank=32,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
        seed=seed,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop=[tokenizer.eos_token] if tokenizer.eos_token else None,
        seed=seed,
    )
    lora = LoRARequest("sccdec", 1, adapter)
    generated = llm.generate(
        [initial_prompts[index] for index in initial_eligible],
        params,
        lora_request=lora,
    )
    initial_sources = [""] * len(functions)
    initial_finish_reasons: list[str | None] = ["context_overflow"] * len(functions)
    for index, generation in zip(initial_eligible, generated):
        initial_sources[index] = generation.outputs[0].text
        initial_finish_reasons[index] = getattr(
            generation.outputs[0], "finish_reason", None
        )

    with ThreadPoolExecutor(max_workers=compile_workers) as executor:
        reconstructed = list(
            executor.map(
                lambda source: (
                    compile_candidate_to_assembly(source, optimization)
                    if source
                    else None
                ),
                initial_sources,
            )
        )

    final_sources = list(initial_sources)
    second_indices: list[int] = []
    second_prompts: list[str] = []
    for index, (function, initial_source, context_assembly) in enumerate(
        zip(functions, initial_sources, reconstructed)
    ):
        if not initial_source or context_assembly is None:
            continue
        messages = [
            {"role": "user", "content": direct_prompt(context_assembly)},
            {"role": "assistant", "content": extract_c_code(initial_source)},
            {"role": "user", "content": direct_prompt(function.assembly)},
        ]
        prompt = chat_prompt(tokenizer, messages)
        if len(tokenizer(prompt, add_special_tokens=True)["input_ids"]) <= token_limit:
            second_indices.append(index)
            second_prompts.append(prompt)
    if second_prompts:
        second_generated = llm.generate(
            second_prompts, params, lora_request=lora
        )
        for index, generation in zip(second_indices, second_generated):
            final_sources[index] = generation.outputs[0].text

    second_set = set(second_indices)
    records: list[dict[str, Any]] = []
    for index, function in enumerate(functions):
        records.append(
            {
                "function_name": function.name,
                "source": restore_function_name(
                    final_sources[index], function.name
                ),
                "initial_source": restore_function_name(
                    initial_sources[index], function.name
                ),
                "error": None if initial_sources[index] else "context_overflow",
                "assembly": function.assembly,
                "self_context_assembly": reconstructed[index],
                "self_context_used": index in second_set,
                "generation": {
                    "initial_prompt_tokens": initial_lengths[index],
                    "initial_finish_reason": initial_finish_reasons[index],
                },
            }
        )
    return binary_path, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--optimization",
        choices=("O0", "O1", "O2", "O3"),
        default="O3",
        help="Optimization used to build the input binary (needed by SCC).",
    )
    parser.add_argument("--no-one-shot", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--compile-workers", type=int, default=8)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs to reduce extra memory while sharing a GPU.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    names = read_function_names(args)
    binary, records = decompile(
        args.binary,
        names,
        base_model=args.base_model,
        adapter=args.adapter,
        optimization=args.optimization,
        one_shot=not args.no_one_shot,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        compile_workers=args.compile_workers,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    write_results(
        output=args.output,
        method="sc2dec",
        binary=binary,
        records=records,
        configuration={
            "base_model": args.base_model,
            "adapter": args.adapter,
            "optimization": args.optimization,
            "one_shot": not args.no_one_shot,
            "self_constructed_context": True,
            "decoder": "greedy",
            "max_model_len": args.max_model_len,
            "max_new_tokens": args.max_new_tokens,
            "enforce_eager": args.enforce_eager,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
