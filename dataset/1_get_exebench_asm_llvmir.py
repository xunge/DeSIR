import argparse
import json
import os
import asyncio
import re
import tempfile
from asyncio import subprocess
from pathlib import Path
from tqdm import tqdm

os.environ["USE_TORCH"] = "FALSE"
from datasets import load_dataset
from multiprocessing import Pool

from ir_normalization import normalize_ir_legacy, normalize_ir_safe


def extract_useful_ir_with_c_code(input_file, source_file, is_merge=False):
    """Compatibility wrapper for the normalization used by old checkpoints."""

    del source_file, is_merge
    return normalize_ir_legacy(Path(input_file).read_text(encoding="utf-8"))


async def compile_c_to_bin(
        tmpdir, formatted_code_file, optimization, compiler="gcc"
):
    binary_file = os.path.join(tmpdir, f"code{optimization}-{compiler}.out")

    try:
        proc_compile = await asyncio.create_subprocess_exec(
            f"{compiler}",
            "-shared",
            "-fPIC",
            "-g3",
            optimization,
            "-o",
            binary_file,
            formatted_code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.wait_for(proc_compile.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the process...")
        proc_compile.terminate()
        try:
            await asyncio.wait_for(proc_compile.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing it...")
            proc_compile.kill()
            await proc_compile.wait()

    if not os.path.exists(binary_file):
        raise ValueError("Binary not found!")
    return binary_file


async def compile_c_to_ll(
        tmpdir, formatted_code_file, optimization
):
    ll_file = os.path.join(tmpdir, f"code{optimization}.ll")

    try:
        proc_compile = await asyncio.create_subprocess_exec(
            # "gcc",
            "clang",
            "-S",
            "-emit-llvm",
            "-g3",
            optimization,
            "-o",
            ll_file,
            formatted_code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        await asyncio.wait_for(proc_compile.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("Timeout reached, terminating the process...")
        proc_compile.terminate()
        try:
            await asyncio.wait_for(proc_compile.wait(), timeout=2)
        except asyncio.TimeoutError:
            print("Process did not terminate, killing it...")
            proc_compile.kill()
            await proc_compile.wait()

    raw_ll = Path(ll_file).read_text(encoding="utf-8")
    return {
        "legacy": normalize_ir_legacy(raw_ll),
        "safe": normalize_ir_safe(raw_ll),
        "raw": raw_ll,
    }


async def decompile_bin_to_asm(
        tmpdir, binary_file, function_name
):
    proc_decompile = await asyncio.create_subprocess_exec(
        "objdump",
        "-d",
        "-S",
        "--source-comment=;",
        binary_file,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    asm, _ = await proc_decompile.communicate()
    asm = asm.decode("utf-8")

    function_name_chk = f"<{function_name}>:"

    if function_name_chk not in asm:
        raise ValueError("Function not found in asm!")

    asm = function_name_chk + asm.split(function_name_chk)[-1].split("\n\n")[0]
    asm_sp = asm.split("\n")

    code_blocks = []
    asm_blocks = []

    last_code = False
    for tmp in asm_sp:
        if tmp.startswith(";"):
            if last_code:
                code_blocks[-1] += tmp + "\n"
            else:
                code_blocks.append(tmp + "\n")
            last_code = True
            continue

        if len(tmp.split("\t")) < 3 and "00" in tmp:
            continue
        idx = min(len(tmp.split("\t")) - 1, 2)
        tmp_asm = "\t".join(tmp.split("\t")[idx:])
        tmp_asm = tmp_asm.split("#")[0].strip()

        if last_code:
            asm_blocks.append(tmp_asm + "\n")
        elif not code_blocks:
            asm_blocks.append(tmp_asm + "\n")
        else:
            asm_blocks[-1] += tmp_asm + "\n"
        last_code = False

    asm_clean = ''.join(asm_blocks)
    return asm_clean


async def compile_with_optimization(
        tmpdir, formatted_code_file, function_name, optimization
):
    binary_gcc_file = await compile_c_to_bin(tmpdir, formatted_code_file, optimization, compiler="gcc")
    binary_clang_file = await compile_c_to_bin(tmpdir, formatted_code_file, optimization, compiler="clang")
    ir_variants = await compile_c_to_ll(tmpdir, formatted_code_file, optimization)
    asm_gcc_clean = await decompile_bin_to_asm(tmpdir, binary_gcc_file, function_name)
    asm_clang_clean = await decompile_bin_to_asm(tmpdir, binary_clang_file, function_name)

    return optimization, asm_clang_clean, asm_gcc_clean, ir_variants


async def compile(idx, synth_deps, function_def, function_name):
    function_def, remain = function_def.split("{", maxsplit=1)
    function_def = (
        function_def.replace("static", "")
        .replace("inline", "")
        .replace("\n", " ")
        .strip()
    )
    remain, right_bracket = remain.rsplit("}", maxsplit=1)
    remain = re.sub(r"#\s+\d+\s+\"[^\"]+\"", "", remain)
    function_def += " {" + remain + "\n}"

    function_def = re.sub("\n+", "\n", function_def)

    if synth_deps is not None:
        full_code = synth_deps.strip() + "\n" + function_def.strip() + "\n\n"
    else:
        full_code = function_def.strip() + "\n\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, "code.c")
        formatted_code_file = os.path.join(tmpdir, "formatted_code.c")
        with open(code_file, "w") as f:
            f.write(full_code)

        proc_compile = await asyncio.create_subprocess_exec(
            "clang-format",
            code_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        formatted_code, _ = await proc_compile.communicate()
        formatted_code = formatted_code.decode("utf-8")

        with open(formatted_code_file, "w") as f:
            f.write(formatted_code)

        tasks = [
            asyncio.create_task(
                compile_with_optimization(
                    tmpdir, formatted_code_file, function_name, optimization
                )
            )
            for optimization in ["-O0", "-O1", "-O2", "-O3"]
        ]

        asm_codes = {}
        for example in asyncio.as_completed(tasks):
            try:
                optimization, asm_clang, asm_gcc, ir_variants = await example
                if asm_gcc is not None and asm_clang is not None:
                    asm_codes[f"{optimization}_clang"] = asm_clang
                    asm_codes[f"{optimization}_gcc"] = asm_gcc
                    # ``_ll`` remains the legacy field for released-model
                    # reproducibility.  New experiments should select
                    # ``_ll_safe`` explicitly.
                    asm_codes[f"{optimization}_ll"] = ir_variants["legacy"]
                    asm_codes[f"{optimization}_ll_safe"] = ir_variants["safe"]
                    asm_codes[f"{optimization}_ll_raw"] = ir_variants["raw"]
            except Exception as e:
                pass

        if len(asm_codes) == 0:
            raise ValueError("No asm code generated!")

        asm_codes["idx"] = idx
        asm_codes["synth_deps"] = synth_deps
        asm_codes["function_def"] = function_def
        asm_codes["function_name"] = function_name
        asm_codes["formatted_code"] = formatted_code
        return asm_codes


def run_process_batch(args):
    asyncio.run(process_batch(*args))


async def process_batch(start_idx, num, data_dir, split, output):
    dataset = load_dataset(data_dir, split=split, trust_remote_code=True)
    total_examples = len(dataset)

    end_idx = min(start_idx + num, total_examples)

    batch_dataset = dataset.select(range(start_idx, end_idx), keep_in_memory=True)
    print(f"Processing range: {start_idx}-{end_idx}, total: {len(batch_dataset)}")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        pbar = tqdm(enumerate(batch_dataset), total=len(batch_dataset))
        for idx, item in pbar:
            try:
                code_asm = await compile(
                    idx=start_idx + idx,
                    synth_deps=item["synth_deps"],
                    function_def=item["func_def"],
                    function_name=item["fname"],
                )
                if code_asm is not None:
                    f.write(json.dumps(code_asm) + "\n")
            except Exception as e:
                pbar.set_postfix_str(f"Error[{start_idx + idx}]: {e}")


def process_batches_in_parallel(start_idx, num, batch_size, data_dir, split, output_template, workers):
    total_batches = (num + batch_size - 1) // batch_size
    print('total_batches:', total_batches)
    tasks = []

    for batch_idx in range(total_batches):
        batch_start = start_idx + batch_idx * batch_size
        batch_output = output_template.replace("start", str(batch_start))

        if os.path.exists(batch_output):
            print(f"File {batch_output} already exists, skipping...")
            continue

        tasks.append((batch_start, batch_size, data_dir, split, batch_output))

    with Pool(processes=min(workers, max(1, len(tasks)))) as pool:
        pool.map(run_process_batch, tasks)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start_idx", type=int, default=0, required=False)
    parser.add_argument("-n", "--num", type=int, default=885074, required=False)
    parser.add_argument("--batch_size", type=int, default=30000, required=False)
    parser.add_argument("--data_dir", type=str, default="/home/jiang/dataset/exebench/", required=False)
    parser.add_argument("--split", type=str, default="train_real_compilable", required=False)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output template containing the token 'start', e.g. data/train-start.jsonl",
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if "start" not in args.output:
        parser.error("--output must contain the token 'start' to make batch filenames unique")

    process_batches_in_parallel(
        start_idx=args.start_idx,
        num=args.num,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        split=args.split,
        output_template=args.output,
        workers=args.workers,
    )


if __name__ == "__main__":
    asyncio.run(main())
