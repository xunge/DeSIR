# Review-driven experiment protocol

The released-checkpoint replication, Task-1 budget ablation, differential
tests, SLaDe/Nova/sc²dec artifact evaluation, and held-out ExeBench evaluation
were run on 2026-07-23. The MBPP-Decompile cross-benchmark experiment was
completed on 2026-07-25 and extended to DecIR-6.7B on 2026-07-26; the
matched-Goron HumanEval experiment was completed on 2026-07-26, and the
IDA Pro 9.0/Hex-Rays and Ghidra 10.4 GCC HumanEval experiments were
completed on 2026-07-27. The official Nova-6.7B-BCR HumanEval replication
was completed on 2026-07-28.
The raw
predictions and metric JSON files are under
`results/`. The larger safe/raw/legacy/shuffled three-seed retraining
protocol below remains a registered experiment; never replace its paper cells
with guessed values.

Run commands from the `project` directory unless a block says otherwise.

## Executed results

The local checkpoints used were:

- DecIR-1.3B: `/home/jiang/model/DecIR-1.3B`
- DecIR-6.7B: `/home/jiang/model/decir-6.7b`
- LLM4Decompile-1.3B-v1.5:
  `/home/jiang/model/llm4decompile-1.3b-v1.5`
- LLM4Decompile-6.7B-v1.5:
  `/home/jiang/model/llm4decompile-6.7b-v1.5`
- sc²dec adapter: `/home/jiang/model/sccdec-lora`
- Nova-1.3B-BCR: `/home/jiang/model/nova-1.3b-bcr`
- Nova-6.7B-BCR: `/home/jiang/model/nova-6.7b-bcr`
- LLM4Decompile-1.3B-v1.6 (Decompile-Bench C/C++):
  `/home/jiang/model/llm4decompile-1.3b-v1.6`
- ReF Decompile merged 6.7B checkpoint:
  `/home/jiang/model/ref-decompile-merged`

Fresh greedy HumanEval-Decompile scores under the legacy GCC compilation
dialect are:

| Input | Model | RC | RE |
|---|---|---:|---:|
| Clang objdump, 656 cases | DecIR-1.3B | 93.45 | 50.00 |
| Clang objdump, 656 cases | LLM4Decompile-1.3B | 84.45 | 20.58 |
| GCC objdump, 656 cases | DecIR-1.3B | 83.54 | 22.10 |
| GCC objdump, 656 cases | LLM4Decompile-1.3B | 89.79 | 31.40 |
| GCC objdump, author predictions | sc²dec-6.7B | 91.62 | 51.52 |
| GCC GAS O0/O3, 328 cases | SLaDe first beam | 43.60 | 19.82 |
| GCC GAS O0/O3, 328 cases | DecIR-1.3B | 52.13 | 2.74 |
| GCC GAS O0/O3, 328 cases | LLM4Decompile-1.3B | 85.37 | 13.72 |

Nova-1.3B's released 20-sample artifact obtains empirical RC/RE Pass@1
85.79/24.93; the completed fresh seed-42 rerun obtains 86.29/24.89
(RE Pass@10 37.68 versus 38.34 in the artifact). Nova-6.7B's released
artifact obtains 88.58/34.80 and its fresh rerun obtains 88.69/34.44
(RE Pass@10 48.22 versus 48.41). SLaDe's five-beam oracle RC/RE is
45.73/23.78. The separately built PsycheC stage attempted 219 of 1,640 beam
candidates but repaired none on this benchmark.

The requested HumanEval-Decompile C-only mixed-input-compiler table uses
Clang objdump inputs for DecIR, GCC assembly for the neural baselines, and
fresh GCC executables for IDA Pro 9.0 and Ghidra 10.4. Each cell is RC/RE
(%), every O-level contains 164 problems, and generated candidates use the
same GCC GNU C harness:

| Model | Input CC | O0 | O1 | O2 | O3 | Average | Avg. Edit |
|---|---|---:|---:|---:|---:|---:|---:|
| DecIR-1.3B | Clang | 96.95/74.39 | 92.68/42.07 | 92.07/41.46 | 92.07/42.07 | 93.45/50.00 | 47.45 |
| DecIR-6.7B | Clang | 96.95/76.22 | 93.90/52.44 | 91.46/49.39 | 90.85/50.61 | 93.29/57.16 | 47.72 |
| LLM4Decompile-1.3B-v1.5 | GCC | 90.85/50.61 | 90.24/25.61 | 90.24/25.00 | 87.80/24.39 | 89.79/31.40 | 44.15 |
| LLM4Decompile-6.7B-v1.5 | GCC | 93.90/68.29 | 92.68/43.29 | 93.90/40.85 | 92.68/37.80 | 93.29/47.56 | 47.56 |
| sc²dec-6.7B, corrected local final | GCC | 81.71/61.59 | 84.76/45.73 | 82.32/38.41 | 82.32/37.20 | 82.77/45.73 | 50.85 |
| SLaDe first beam | GCC | 46.34/31.71 | 50.61/10.98† | 48.17/10.37† | 40.85/7.93 | 46.49/15.24 | 29.28 |
| Nova-1.3B, first sample | GCC | 84.76/37.20 | 85.98/19.51 | 89.02/21.34 | 87.20/17.07 | 86.74/23.78 | 45.96 |
| Nova-6.7B, first sample | GCC | 93.29/49.39 | 90.24/31.71 | 87.20/31.10 | 85.37/25.00 | 89.02/34.30 | 47.52 |
| IDA Pro 9.0 (Hex-Rays) | GCC binary | 98.17/87.20 | 98.17/80.49 | 93.29/70.73 | 89.02/69.51 | 94.66/76.98 | 27.83 |
| Ghidra 10.4 | GCC binary | 98.17/76.83 | 96.34/67.68 | 95.73/64.02 | 89.63/61.59 | 94.97/67.53 | 22.53 |

† SLaDe publishes only x86 O0/O3 checkpoints; O1/O2 use its O3 checkpoint
zero-shot. The sc²dec row is the output of the local released FAE LoRA using
the paper-consistent ordinary `alpha/r = 2` scale and self-constructed-context
second pass, rather than the separate author prediction artifact. Edit uses
whitespace-normalized character-level
Levenshtein similarity against the stored reference C field. The generated
JSON/CSV/Markdown comparison is
`results/metrics/humaneval-c-mixed-compiler-comparison.*`.

Nova-6.7B's O0/O1/O2/O3 Edit similarities are
53.21/46.80/45.59/44.50. Reproduce and re-summarize its official
20-candidate seed-42 run with:

```bash
python evaluation/humaneval/nova_6_7b.py
python evaluation/humaneval/summarize_nova_6_7b.py
```

For the IDA row, `/home/jiang/ida-pro-9.0/idat` decompiles an unstripped
executable compiled with `gcc -O* ... -lm`. No debug information or reference
source signature/type is supplied. Deterministic postprocessing removes IDA
calling-convention annotations, maps IDA primitive/SIMD syntax to GNU C, and
restores referenced globals from IDA's binary image. The raw pseudocode,
binary-derived global records, and normalized candidate are all retained in
`results/predictions/ida-pro-9.0-hexrays-gcc.json`. Per-level Edit is
35.08/26.82/25.96/23.44 for O0/O1/O2/O3. Exact commands are documented in
`evaluation/humaneval/README.md`.

For the Ghidra row,
`/home/jiang/ghidra_10.4_PUBLIC/support/analyzeHeadless` analyzes the same
fresh, unstripped, no-debug GCC executables using Java 17. No reference
signature or source type is applied. The deterministic adapter maps Ghidra
primitive/helper and exact-width partial-scalar syntax to GNU C and restores
only globals read from Ghidra's binary memory image. Its O0/O1/O2/O3 Edit is
25.19/23.05/22.22/19.67. Raw pseudocode and normalized candidates are in
`results/predictions/ghidra-10.4-gcc.json`.

The released DecIR Task-1 output-budget ablation is:

| Max new tokens | Exact marker | Truncated | IR valid | C compile | C RE |
|---:|---:|---:|---:|---:|---:|
| 1024 | 48.32 | 58.23 | 26.07 | 40.55 | 23.02 |
| 2048 | 80.34 | 25.30 | 31.40 | 69.51 | 36.13 |
| 4096 | 94.97 | 7.93 | 32.47 | 84.91 | 42.07 |

On 116/656 scalar-signature cases eligible for 1,000-input differential
testing, DecIR passes 55.17% and LLM4Decompile passes 32.76%. The following is
the earlier structural-only diagnostic on 1,606 valid GCC-O3 functions in the
local ExeBench `test_real` Parquet partition:

| Model | Compile | Token edit | Control F1 | Operator F1 | Signature exact |
|---|---:|---:|---:|---:|---:|
| DecIR-1.3B | 70.42 | 45.72 | 68.59 | 61.57 | 11.77 |
| LLM4Decompile-1.3B | 73.97 | 52.06 | 71.23 | 71.53 | 7.10 |

The confirmatory ExeBench experiment now uses the official v1.01
`test_real.tar.gz`, validates each reference against all ten `real_io_pairs`,
and reports executable O0--O3 results for 1,941 C functions per level. Its
commands and protocol are in
[`evaluation/exebench/README.md`](evaluation/exebench/README.md); the final
machine-readable and Markdown comparisons are written under
`results/metrics/exebench-c-model-comparison.*`.

Each ExeBench cell below is compilation/strict function accuracy (%); strict
accuracy requires all ten official tests to pass:

| Model | O0 | O1 | O2 | O3 | Average |
|---|---:|---:|---:|---:|---:|
| DecIR-1.3B | 55.23/19.06 | 71.97/34.00 | 69.81/31.38 | 69.24/31.12 | 66.56/28.89 |
| DecIR-6.7B | 90.11/58.01 | 90.26/59.45 | 89.34/55.80 | 89.13/54.46 | 89.71/56.93 |
| LLM4Decompile-1.3B-v1.5 | 73.00/31.38 | 72.49/32.77 | 69.96/30.09 | 68.68/29.83 | 71.03/31.01 |
| LLM4Decompile-6.7B-v1.5 | 81.61/58.89 | 75.32/47.66 | 73.83/45.44 | 72.85/44.36 | 75.90/49.09 |
| sc²dec-6.7B, corrected final | 86.09/62.49 | 85.78/52.40 | 83.87/49.67 | 83.10/49.30 | 84.71/53.46 |
| SLaDe first beam | 44.67/36.94 | 44.98/22.21† | 44.87/23.03† | 42.30/22.10 | 44.20/26.07 |
| Nova-1.3B, first sample | 62.60/35.50 | 59.76/30.60 | 55.38/29.11 | 55.28/27.41 | 58.26/30.65 |

† SLaDe O1/O2 use its released O3 checkpoint zero-shot. Nova uses the
official address-aware preprocessing and one fixed-seed sampled candidate.

### Requested mixed-input-compiler C-only comparison

On 2026-07-28, Clang 18.1.3 regenerated the C-only model inputs at O0--O3.
MBPP functions are linked with a synthetic driver before GNU objdump extracts
the target function. This resolves all 5,258 call targets; the earlier
unlinked-object artifact lost 5,248 relocation-backed symbols and is retained
only for diagnosis. Candidate execution retained the common GCC/g++ 13.3.0
oracle so the rerun isolates input code generation. ExeBench retains 1,933
functions per level after four-level assembly construction and official-I/O
reference validation; MBPP-Decompile retains 973 problems per level after
excluding one source that uses GCC's nested-function extension. The five
baseline rows retain GCC 13.3.0 inputs, with 1,941 ExeBench functions and 974
MBPP problems per level. The exact commands are in the ExeBench and MBPP
evaluation READMEs. The generated comparison, including per-level Edit
values, is
`results/metrics/mixed-compiler-c-model-comparison.{json,csv,md}`.

ExeBench cells are compilation/strict function accuracy (%); Edit uses the
same whitespace-normalized Levenshtein definition as MBPP:

| Model | Input CC | N/level | O0 | O1 | O2 | O3 | Average | Avg. Edit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DecIR-1.3B | Clang | 1,933 | 75.27/41.54 | 78.53/41.44 | 78.38/40.77 | 78.17/40.56 | 77.59/41.08 | 44.19 |
| DecIR-6.7B | Clang | 1,933 | 91.15/70.82 | 91.93/61.98 | 91.26/60.27 | 91.26/60.01 | 91.40/63.27 | 56.69 |
| LLM4Decompile-1.3B-v1.5 | GCC | 1,941 | 73.00/31.38 | 72.49/32.77 | 69.96/30.09 | 68.68/29.83 | 71.03/31.01 | 47.88 |
| LLM4Decompile-6.7B-v1.5 | GCC | 1,941 | 81.61/58.89 | 75.32/47.66 | 73.83/45.44 | 72.85/44.36 | 75.90/49.09 | 53.38 |
| sc²dec-6.7B, corrected | GCC | 1,941 | 86.09/62.49 | 85.78/52.40 | 83.87/49.67 | 83.10/49.30 | 84.71/53.46 | 53.57 |
| SLaDe | GCC | 1,941 | 44.67/36.94 | 44.98/22.21† | 44.87/23.03† | 42.30/22.10 | 44.20/26.07 | 43.19 |
| Nova-1.3B | GCC | 1,941 | 62.60/35.50 | 59.76/30.60 | 55.38/29.11 | 55.28/27.41 | 58.26/30.65 | 39.13 |
| Nova-6.7B | GCC | 1,941 | 66.56/41.52 | 63.37/36.53 | 59.56/33.59 | 58.32/32.30 | 61.95/35.99 | 39.81 |

MBPP-Decompile cells are RC/RE (%):

| Model | Input CC | N/level | O0 | O1 | O2 | O3 | Average | Avg. Edit |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DecIR-1.3B | Clang | 973 | 87.36/64.85 | 83.04/40.60 | 82.43/38.03 | 82.43/38.23 | 83.81/45.43 | 50.57 |
| DecIR-6.7B | Clang | 973 | 89.31/66.91 | 85.61/48.20 | 84.99/46.04 | 84.58/44.60 | 86.13/51.44 | 52.30 |
| LLM4Decompile-1.3B-v1.5 | GCC | 974 | 80.49/47.84 | 72.48/27.93 | 70.43/27.00 | 71.56/26.28 | 73.74/32.26 | 50.93 |
| LLM4Decompile-6.7B-v1.5 | GCC | 974 | 83.06/61.09 | 77.41/43.53 | 76.69/43.33 | 77.31/42.81 | 78.62/47.69 | 53.01 |
| sc²dec-6.7B, corrected | GCC | 974 | 78.75/58.62 | 76.59/41.99 | 75.56/41.17 | 77.62/40.25 | 77.13/45.51 | 58.56 |
| SLaDe | GCC | 974 | 46.92/33.47 | 53.18/18.07† | 48.87/18.89† | 43.12/16.53 | 48.02/21.74 | 39.46 |
| Nova-1.3B | GCC | 974 | 69.30/36.96 | 66.32/24.95 | 64.58/24.44 | 64.58/21.77 | 66.20/27.03 | 50.10 |
| Nova-6.7B | GCC | 974 | 71.15/43.33 | 66.32/32.55 | 66.53/31.72 | 64.78/28.75 | 67.20/34.09 | 51.48 |

† SLaDe O1/O2 apply its released O3 checkpoint zero-shot. Nova reports its
first fixed-seed sampled candidate rather than Pass@k.

The completed MBPP-Decompile evaluation excludes only the 32 instances whose
upstream reference translations fail their own tests. Direct models use the
same 7,760 raw-assembly inputs and decoder:

| Model | Scope | RC | RE | Edit |
|---|---|---:|---:|---:|
| DecIR-1.3B | C/C++, O0--O3 | 41.53 | 15.03 | 27.85 |
| DecIR-6.7B | C/C++, O0--O3 | 44.32 | 21.56 | 31.20 |
| LLM4Decompile-1.3B-v1.5 | C/C++, O0--O3 | 47.07 | 21.15 | 40.24 |
| LLM4Decompile-1.3B-v1.6 | C/C++, O0--O3 | 63.84 | 26.11 | 49.26 |
| LLM4Decompile-6.7B-v1.5 | C only, O0--O3 | 78.62 | 47.69 | 53.01 |
| sc²dec-6.7B, obsolete rsLoRA diagnostic | C/C++, O0--O3 | 29.43 | 8.70 | 38.04 |
| sc²dec-6.7B, corrected final | C only, O0--O3 | 77.13 | 45.51 | 58.56 |
| Nova-1.3B, first sample | C only, O0--O3 | 66.20 | 27.03 | 50.10 |
| ReF Decompile-6.7B | C only, labelled/tool protocol | 86.86 | 60.27 | 55.72 |
| SLaDe first beam | C only, x86 O0/O3 | 45.02 | 25.00 | 38.94 |
| SLaDe first beam, O3 transfer | C only, O0--O3 | 48.02 | 21.74 | 39.46 |

For the requested C-only comparison, every O-level contains the same 974
underlying MBPP problems. Each cell below is RC/RE (%):

| Model | O0 | O1 | O2 | O3 | Average |
|---|---:|---:|---:|---:|---:|
| DecIR-1.3B | 68.99/27.72 | 68.99/25.15 | 69.71/21.97 | 69.82/20.84 | 69.38/23.92 |
| DecIR-6.7B | 69.61/39.01 | 66.84/30.49 | 68.28/30.80 | 67.04/30.29 | 67.94/32.65 |
| LLM4Decompile-1.3B-v1.5 | 80.49/47.84 | 72.48/27.93 | 70.43/27.00 | 71.56/26.28 | 73.74/32.26 |
| LLM4Decompile-6.7B-v1.5 | 83.06/61.09 | 77.41/43.53 | 76.69/43.33 | 77.31/42.81 | 78.62/47.69 |
| sc²dec-6.7B, corrected final | 78.75/58.62 | 76.59/41.99 | 75.56/41.17 | 77.62/40.25 | 77.13/45.51 |
| SLaDe first beam | 46.92/33.47 | 53.18/18.07† | 48.87/18.89† | 43.12/16.53 | 48.02/21.74 |
| Nova-1.3B, first sample | 69.30/36.96 | 66.32/24.95 | 64.58/24.44 | 64.58/21.77 | 66.20/27.03 |

† SLaDe does not publish x86 O1/O2 checkpoints; these two cells apply its O3
checkpoint zero-shot. O0/O3 retain the released native checkpoint mapping.
Nova uses its official address-aware normalization, and sc²dec is the
dependency-aware second pass, so those systems are protocol-marked rather than
treated as raw-assembly drop-ins.

Scaling DecIR from 1.3B to 6.7B improves RE by 6.53 percentage points
(974-problem cluster-bootstrap 95% CI [5.29, 7.79]). DecIR-6.7B and
LLM4Decompile-v1.5 are statistically tied on RE: +0.41 points
([-0.98, 1.82]), while v1.6 remains 4.55 points stronger ([3.00, 6.08]).
Three independent executions of the fixed DecIR-6.7B predictions give
21.56%--21.60% RE. All variation is confined to three C candidates containing
out-of-bounds array reads; the table retains the conservative 21.56% run.
The paired DecIR-1.3B-minus-v1.5 RE difference is -6.12 percentage points
(974-problem cluster-bootstrap 95% CI [-7.39, -4.86]); by language it is
-8.34 points on C and -3.88 on C++. The v1.6-minus-DecIR difference is
+11.08 points [9.62, 12.53]. ReF invokes its `parse_data` tool on 846/3,896
examples (1,885 calls); one ReF prompt exceeds the common context budget.
SLaDe has 509/1,948 encoder overflows, which are retained as failures.
The earlier C/C++ sc²dec diagnostic used the adapter metadata's rsLoRA flag
and is retained only to document the loading error; it must not be reported
as a reproduced result.  In the corrected C-only run, dependency-aware
self-constructed context is available for 3,163/3,896 rows and changes 380
outputs.  It raises RE from 44.22% to 45.51%; 71 failures become passes and
21 passes regress.  The paired gain is +1.28 points (95% problem-cluster
bootstrap CI [0.69, 1.90]).
Nova's 20-sample execution estimates are Pass@1 27.05 and Pass@10 38.87.
Its author's standalone-candidate RC definition gives Pass@1/Pass@10
76.62/88.42; the 66.20 RC above uses the common Decompile-Bench rule that
links the first sample with the test harness.

These files are the canonical audit trail:

```text
results/predictions/
results/metrics/
decompile-eval/exebench-test-real-gcc-o3.json
baselines/availability.json
```

### DecIR-6.7B-STR on HumanEval-Decompile

The checkpoint `/home/jiang/model/decir-6.7b-str` was evaluated with the
binary-grounded string protocol from
`/home/jiang/projects/DecIR/evaluation_string/decompile_binary.py`. Clang
18.1.3 builds each O0--O3 input binary; referenced strings are represented as
per-function `STR_i` annotations during generation and restored before
scoring. As in the attached batch evaluator, GCC 13.3.0 is the generated-C
compilation and execution oracle.

| Optimization | RC | RE | Edit | TCP |
|---|---:|---:|---:|---:|
| O0 | 92.68 | 83.54 | 45.12 | 85.41 |
| O1 | 92.68 | 60.98 | 37.05 | 70.74 |
| O2 | 92.07 | 59.76 | 36.93 | 68.27 |
| O3 | 93.29 | 60.37 | 37.10 | 68.73 |
| Average | 92.68 | 66.16 | 39.05 | 73.29 |

RC/RE/Edit cover all 656 rows. TCP covers 652 valid reference-oracle rows
(163 per optimization) and uses the function-macro definition. TCP is the
median aggregate run among three executions (73.29/73.28/73.29), limiting
noise from undefined behavior in generated candidates. The prepared
inputs contain 200 string references over 105 rows; 94 model outputs use
`STR_i`, all of which are restored without unresolved tokens. Seventeen
outputs reach the common 1,024-token generation cap, and no prompt exceeds
the 8,192-token context allocation. The command and output paths are in
`evaluation/humaneval/README.md`.

### DecIR-6.7B-STR on MBPP-Decompile and ExeBench

The same binary-grounded STR protocol was run on the balanced Clang C-only
scopes: 973 MBPP functions and 1,933 ExeBench functions at each optimization
level. Generated candidates retain the common GCC/G++ benchmark oracle.

| Benchmark | Opt | N | RC | RE | Edit | TCP |
|---|---:|---:|---:|---:|---:|---:|
| MBPP-Decompile C-only | O0 | 973 | 88.80 | 75.64 | 54.38 | 76.67 |
|  | O1 | 973 | 86.84 | 60.53 | 45.22 | 64.20 |
|  | O2 | 973 | 86.64 | 59.30 | 44.82 | 62.72 |
|  | O3 | 973 | 86.02 | 58.89 | 44.47 | 62.15 |
|  | Average | 3,892 | 87.08 | 63.59 | 47.22 | 66.44 |
| ExeBench C-only | O0 | 1,933 | 93.43 | 83.45 | 54.09 | 84.51 |
|  | O1 | 1,933 | 92.50 | 72.79 | 47.84 | 74.84 |
|  | O2 | 1,933 | 92.08 | 69.89 | 46.56 | 72.27 |
|  | O3 | 1,933 | 92.45 | 70.51 | 46.56 | 72.59 |
|  | Average | 7,732 | 92.62 | 74.16 | 48.76 | 76.05 |

ExeBench RE requires all ten official real I/O pairs to pass; TCP is the
per-function test-pass fraction averaged over functions. Fifty-five ExeBench
prompts exceed the attachment's 7,168-token input budget and are retained as
context-overflow failures. MBPP TCP excludes 16 rows whose reference harness
does not supply a valid runnable oracle (969 functions per optimization);
three executions give 66.42--66.45 and the canonical/middle value is 66.44.
The MBPP/ExeBench prepared inputs contain 1,037/192 referenced strings;
generated code uses `STR_i` in 443/77 rows, and deterministic restoration
leaves no unresolved placeholder in either prediction file.
The full JSON/CSV/Markdown table is
`results/metrics/decir-6.7b-str-clang-mbpp-exebench-summary.*`.

### DecIR-1.3B-STR on three Clang benchmarks

The checkpoint `/home/jiang/model/decir-1.3b-str` uses the same binary-grounded
STR protocol, balanced scopes, context limits, and GCC/G++ candidate oracle as
the 6.7B-STR experiment above.

| Benchmark | Opt | N | RC | RE | Edit | TCP |
|---|---:|---:|---:|---:|---:|---:|
| HumanEval-Decompile | O0 | 164 | 89.02 | 78.05 | 44.96 | 79.85 |
|  | O1 | 164 | 89.02 | 48.17 | 37.70 | 58.86 |
|  | O2 | 164 | 91.46 | 45.73 | 36.42 | 57.65 |
|  | O3 | 164 | 90.24 | 46.95 | 36.44 | 58.71 |
|  | Average | 656 | 89.94 | 54.73 | 38.88 | 63.77 |
| MBPP-Decompile C-only | O0 | 973 | 85.71 | 74.31 | 54.04 | 75.39 |
|  | O1 | 973 | 83.66 | 46.25 | 44.60 | 51.84 |
|  | O2 | 973 | 83.04 | 43.88 | 44.16 | 49.28 |
|  | O3 | 973 | 82.53 | 43.06 | 44.00 | 49.08 |
|  | Average | 3,892 | 83.74 | 51.88 | 46.70 | 56.40 |
| ExeBench C-only | O0 | 1,933 | 93.02 | 81.17 | 53.44 | 82.06 |
|  | O1 | 1,933 | 90.89 | 64.98 | 46.06 | 67.47 |
|  | O2 | 1,933 | 89.65 | 62.70 | 44.90 | 64.95 |
|  | O3 | 1,933 | 89.55 | 62.44 | 44.85 | 64.77 |
|  | Average | 7,732 | 90.78 | 67.82 | 47.31 | 69.81 |

HumanEval and MBPP TCP use the median of three executions: 63.77 and 56.40,
respectively. STR restoration changes 98/430/77 HumanEval/MBPP/ExeBench rows
and leaves no unresolved placeholder. ExeBench retains the same 55 context
overflows as failures. The canonical combined artifacts are
`results/metrics/decir-1.3b-str-clang-three-benchmark-summary.{json,csv,md}`.

### Goron-obfuscated HumanEval-Decompile

The dataset was built on 2026-07-25 and the full evaluation completed on
2026-07-26 with
`/home/jiang/projects/open_source/goron/build/bin/clang` (Clang 7.1.0).
The five passes are evaluated separately, not cumulatively. A sixth `none`
configuration is compiled with the same Goron Clang so that an ordinary
compiler-version change is not mistaken for an obfuscation effect.
Because `-mllvm -irobf-*` is implemented as an LLVM/Clang pass, GCC cannot
produce a Goron-obfuscated binary. All models therefore consume the same
Goron-Clang input artifact; all generated candidates use the common GCC
13.3.0 compilation and execution oracle. Calling the baseline rows “GCC”
refers to that candidate oracle, not to a different, technically unavailable
GCC-obfuscated input.

The canonical Python entry points requested for this experiment are under
`evaluation/humaneval/`; see its `README.md`. The complete run can be
reproduced with:

```bash
python evaluation/humaneval/run_all.py
```

The completed first-candidate results are:

| Model | None RC/RE | INDBR RC/RE | ICALL RC/RE | INDGV RC/RE | CSE RC/RE | CFF RC/RE |
|---|---:|---:|---:|---:|---:|---:|
| DecIR-1.3B | 88.57/41.77 | 33.69/6.10 | 64.02/24.09 | 88.11/38.57 | 85.37/37.96 | 21.80/2.44 |
| DecIR-6.7B | 89.63/50.76 | 39.79/8.54 | 62.04/28.35 | 86.74/48.93 | 84.60/48.17 | 21.49/3.81 |
| LLM4Decompile-1.3B | 82.32/22.56 | 45.58/5.18 | 52.74/13.11 | 79.73/18.90 | 77.59/19.05 | 14.79/2.74 |
| LLM4Decompile-6.7B | 86.43/33.84 | 64.02/10.98 | 52.74/18.60 | 79.12/31.10 | 79.42/32.16 | 17.23/3.51 |
| sc²dec-6.7B, final | 80.18/35.37 | 76.83/16.62 | 58.38/21.80 | 79.27/35.52 | 74.24/33.69 | 31.71/4.88 |
| SLaDe, first beam | 15.55/0.61 | 2.74/0.30 | 8.84/0.61 | 13.72/1.52 | 14.33/2.13 | 2.13/0.30 |
| Nova-1.3B, first sample | 79.88/15.85 | 56.71/3.96 | 68.75/11.74 | 75.76/14.02 | 78.35/12.80 | 41.16/2.29 |
| Nova-6.7B, fixed-batch-seed sample | 84.60/19.51 | 64.02/5.03 | 70.27/15.55 | 80.03/19.21 | 78.96/19.21 | 53.20/3.05 |

SLaDe covers O0/O3 (328 cases per column); the other rows cover O0--O3
(656 cases per column). Across all six configurations, Nova's 20-sample
RC Pass@1/Pass@10 is 66.85/93.02 and RE Pass@1/Pass@10 is 9.96/18.30.
The per-configuration Nova RE Pass@1/Pass@10 values are 14.50/25.20,
4.15/7.72, 11.59/22.53, 13.48/24.81, 13.45/25.15, and 2.59/4.38 in the
table's column order.

The five-obfuscation macro averages (excluding the matched control) are:

| Model | Avg. RC/RE | Avg. Edit | Avg. TCP |
|---|---:|---:|---:|
| DecIR-1.3B | 58.60/21.83 | 30.84 | 27.38 |
| DecIR-6.7B | 58.93/27.56 | 30.32 | 32.58 |
| LLM4Decompile-1.3B-v1.5 | 54.09/11.80 | 29.35 | 18.88 |
| LLM4Decompile-6.7B-v1.5 | 58.51/19.27 | 32.41 | 27.81 |
| sc²dec-6.7B | 64.09/22.50 | 37.82 | 30.85 |
| SLaDe | 8.35/0.97 | 10.41 | 2.37 |
| Nova-1.3B | 64.15/8.96 | 31.87 | 16.18 |
| Nova-6.7B | 69.30/12.41 | 33.92 | 20.70 |

DecIR-6.7B has the highest obfuscated-average RE and TCP. sc²dec has the
highest INDBR and CFF RE, while DecIR-6.7B leads ICALL, INDGV, and CSE.
Nova-6.7B has the highest average RC but substantially lower RE and TCP,
showing that recompilability alone overstates semantic recovery.

Across all 3,936 cases, DecIR-6.7B obtains 64.05% RC and 31.43% RE.
Relative to DecIR-1.3B, the paired RE gain is 6.28 percentage points
(164-problem clustered-bootstrap 95% CI [3.71, 8.87]); the paired RC
difference is 0.46 points (95% CI [-1.96, 2.92]). Under the shared
1,024-token output cap, 1,174/3,936 DecIR-6.7B generations end by length
(508/656 on CFF), with no context-overflow or empty outputs.

Build and reference-test all 164 problems at O0--O3 under all six
configurations:

```bash
python dataset/build_goron_humaneval.py \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --output decompile-eval/humaneval-decompile-goron.json \
  --nova-root ../third_party/nova \
  --clang /home/jiang/projects/open_source/goron/build/bin/clang \
  --workers 32
```

This produces 3,936/3,936 compilable, reference-test-passing executables. The
manifest records the source/output hashes, exact flags, compiler version, and
Nova revision:
`decompile-eval/humaneval-decompile-goron.manifest.json`.

Run DecIR and LLM4Decompile with the same direct prompt and greedy decoder:

```bash
python evaluation/humaneval/decir_6_7b.py --gpu 0
python evaluation/humaneval/llm4decompile_6_7b.py --gpu 1

CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/DecIR-1.3B \
  --dataset decompile-eval/humaneval-decompile-goron.json \
  --output results/predictions/decir-1.3b-humaneval-goron.json \
  --mode direct --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 16384 --max-new-tokens 1024 \
  --temperature 0 --top-p 1 --seed 42

CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/llm4decompile-1.3b-v1.5 \
  --dataset decompile-eval/humaneval-decompile-goron.json \
  --output \
    results/predictions/llm4decompile-1.3b-v1.5-humaneval-goron.json \
  --mode direct --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 16384 --max-new-tokens 1024 \
  --temperature 0 --top-p 1 --seed 42
```

Run the released sc²dec LoRA over LLM4Decompile-6.7B-v1.5. `--one-shot`
adds the authors' fixed source/assembly demonstration; unless
`--skip-self-context` is supplied, the runner then performs one
self-constructed-context round after the initial candidate. The released
adapter metadata sets `use_rslora=true`, but the paper and LlamaFactory
training configuration specify ordinary LoRA with rank 32 and alpha 64.
Therefore the corrected replication uses `--lora-scaling standard`, giving
the paper-consistent scale `alpha/r = 2.0` while leaving the checkpoint
weights unchanged:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
python evaluation/run_inference_sccdec.py \
  --base-model /home/jiang/model/llm4decompile-6.7b-v1.5 \
  --adapter /home/jiang/model/sccdec-lora --tokenizer-source base \
  --lora-scaling standard \
  --dataset decompile-eval/humaneval-decompile-goron.json \
  --output \
    results/predictions/sccdec-6.7b-standard-lora-humaneval-goron.json \
  --one-shot --tensor-parallel-size 1 --gpu-memory-utilization 0.9 \
  --max-model-len 16384 --max-new-tokens 1024 \
  --compile-workers 32 --seed 42
```

SLaDe's released x86 checkpoints consume compiler-emitted GAS rather than
objdump text and publish only O0/O3 models. Generate that native input with
the identical source, optimization level, and Goron flag, then retain O1/O2
as explicitly out of scope:

```bash
python dataset/build_goron_slade_inputs.py \
  --dataset decompile-eval/humaneval-decompile-goron.json \
  --output decompile-eval/humaneval-decompile-goron-slade-gas.json \
  --clang /home/jiang/projects/open_source/goron/build/bin/clang \
  --workers 32

CUDA_VISIBLE_DEVICES=0 .venv-slade/bin/python \
evaluation/run_inference_slade.py \
  --dataset decompile-eval/humaneval-decompile-goron-slade-gas.json \
  --output results/predictions/slade-humaneval-goron-gas.json \
  --o0-model \
    ../third_party/slade_artifact/output/export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best \
  --o3-model \
    ../third_party/slade_artifact/output/export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best \
  --device cuda:0 --batch-size 8 --num-beams 5 \
  --num-return-sequences 5 --max-new-tokens 512

python evaluation/apply_slade_type_inference.py \
  --predictions results/predictions/slade-humaneval-goron-gas.json \
  --output results/predictions/slade-humaneval-goron-gas-typed.json \
  --slade-root ../third_party/slade_artifact \
  --candidate-field candidates --output-field typed_candidates --workers 32

python evaluation/evaluate_sampled_predictions.py \
  --predictions results/predictions/slade-humaneval-goron-gas-typed.json \
  --candidate-field typed_candidates --k 1 5 --ordered-candidates \
  --compiler gcc --workers 32 --no-strict \
  --output results/metrics/slade-humaneval-goron-gas-typed-beams.json
```

Nova uses the authors' hierarchical-attention model and address-aware
normalization. Its official 20-sample, 1024-output-token setting is retained.
The 20 candidates are generated in two batches of 10 to bound KV-cache
memory; independent shards use absolute dataset-index seeds, so merging does
not change the samples:

```bash
evaluation/run_nova_goron_shards.sh 0 \
  cff-b:3608:3936 none-b:328:656 indbr-b:984:1312 \
  icall-b:1640:1968 indgv-b:2296:2624 cse-b:2952:3280

evaluation/run_nova_goron_shards.sh 1 \
  cff-a:3280:3608 none-a:0:328 indbr-a:656:984 \
  icall-a:1312:1640 indgv-a:1968:2296 cse-a:2624:2952
```

The corresponding Nova-6.7B two-GPU entry point uses one fixed-batch-seed
sample per input (input batch 8, temperature 0.2, top-p 0.95), a 1,024-token
output limit, and addressed Goron input. Shard ranges and batch starts
determine the recorded seeds. This is the candidate used by the
RC/RE/Edit/TCP table; generating another 19 samples would not change those
single-sample metrics:

```bash
python evaluation/humaneval/nova_goron_6_7b.py \
  --gpu-a 0 --gpu-b 1
```

Nova-1.3B retains the official 20-sample protocol. Merge its twelve shards
and score the first candidate plus all 20 samples:

```bash
python evaluation/merge_prediction_shards.py \
  --dataset decompile-eval/humaneval-decompile-goron.json \
  --inputs \
    results/predictions/nova-1.3b-humaneval-goron-none-a.json \
    results/predictions/nova-1.3b-humaneval-goron-none-b.json \
    results/predictions/nova-1.3b-humaneval-goron-indbr-a.json \
    results/predictions/nova-1.3b-humaneval-goron-indbr-b.json \
    results/predictions/nova-1.3b-humaneval-goron-icall-a.json \
    results/predictions/nova-1.3b-humaneval-goron-icall-b.json \
    results/predictions/nova-1.3b-humaneval-goron-indgv-a.json \
    results/predictions/nova-1.3b-humaneval-goron-indgv-b.json \
    results/predictions/nova-1.3b-humaneval-goron-cse-a.json \
    results/predictions/nova-1.3b-humaneval-goron-cse-b.json \
    results/predictions/nova-1.3b-humaneval-goron-cff-a.json \
    results/predictions/nova-1.3b-humaneval-goron-cff-b.json \
  --output results/predictions/nova-1.3b-humaneval-goron.json

python evaluation/evaluate_predictions.py \
  --predictions results/predictions/nova-1.3b-humaneval-goron.json \
  --output results/metrics/nova-1.3b-humaneval-goron-first-sample.json \
  --compiler gcc --workers 64 --no-strict \
  --bootstrap-samples 10000 --seed 42

python evaluation/evaluate_sampled_predictions.py \
  --predictions results/predictions/nova-1.3b-humaneval-goron.json \
  --candidate-field candidates --k 1 10 --compiler gcc \
  --workers 64 --no-strict \
  --output results/metrics/nova-1.3b-humaneval-goron-pass-at-k.json
```

All first-candidate files, including sc²dec's `initial_output` diagnostic,
are scored with the same command shape:

```bash
python evaluation/evaluate_predictions.py \
  --predictions PREDICTIONS.json --output METRICS.json \
  --compiler gcc --workers 32 --no-strict \
  --bootstrap-samples 10000 --seed 42
```

Finally, export the canonical JSON, CSV, and paper table:

```bash
python evaluation/aggregate_goron_results.py \
  --decir results/metrics/decir-1.3b-humaneval-goron.json \
  --decir-6.7b results/metrics/decir-6.7b-humaneval-goron.json \
  --llm4decompile \
    results/metrics/llm4decompile-1.3b-v1.5-humaneval-goron.json \
  --llm4decompile-6.7b \
    results/metrics/llm4decompile-6.7b-v1.5-humaneval-goron.json \
  --sccdec \
    results/metrics/sccdec-6.7b-standard-lora-humaneval-goron-final.json \
  --slade results/metrics/slade-humaneval-goron-gas-typed-beams.json \
  --nova-first \
    results/metrics/nova-1.3b-humaneval-goron-first-sample.json \
  --nova-sampled \
    results/metrics/nova-1.3b-humaneval-goron-pass-at-k.json \
  --nova-6.7b-first \
    results/metrics/nova-6.7b-humaneval-goron-first-sample.json \
  --nova-6.7b-sampled \
    results/metrics/nova-6.7b-humaneval-goron-pass-at-k.json \
  --output-json results/metrics/humaneval-goron-summary.json \
  --output-csv results/metrics/humaneval-goron-summary.csv \
  --output-tex ../paper/generated_goron_table.tex

python evaluation/humaneval/tcp.py
python evaluation/humaneval/summarize_goron_all_models.py
```

### MBPP-Decompile cross-benchmark evaluation

Import the authors' MBPP split, preserving each source problem as a confidence
interval cluster:

```bash
python dataset/import_decompile_bench.py \
  --input ../third_party/LLM4Decompile/decompile-bench/data/mbpp-decompile.json \
  --output decompile-eval/mbpp-decompile.json \
  --comparison-dataset decompile-eval/decompile-eval-executable-clang-obj.json
```

The adjacent manifest pins the upstream file by SHA-256 and records zero exact
whitespace-normalized source overlap with HumanEval-Decompile. Validate the
reference functions before inspecting model predictions. On this
host, `libcrypto.so.3` is installed but the development-package symlink
`libcrypto.so` is not, so the exact installed ABI is selected explicitly:

```bash
python evaluation/evaluate_decompile_bench.py \
  --predictions decompile-eval/mbpp-decompile.json \
  --prediction-field c_func \
  --output results/metrics/mbpp-decompile-reference-validation.json \
  --workers 64 --bootstrap-samples 10000 --seed 42 \
  '--crypto-library=-l:libcrypto.so.3'
```

The reference audit passes 7,760/7,792 instances. The 32 failures correspond
to eight invalid upstream C++ translations at four optimization levels; they
are excluded for every model through `--reference-validation`.

Generate DecIR and LLM4Decompile predictions with the same prompt, context,
output budget, greedy decoder, and seed:

```bash
python evaluation/mbpp/decir_6_7b.py --gpus 0,1

CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/DecIR-1.3B \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/decir-1.3b-mbpp-decompile.json \
  --gpu-memory-utilization 0.20 --max-model-len 16384 \
  --max-new-tokens 1024 --temperature 0 --seed 42

CUDA_VISIBLE_DEVICES=1 python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/llm4decompile-1.3b-v1.5 \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/llm4decompile-1.3b-v1.5-mbpp-decompile.json \
  --gpu-memory-utilization 0.20 --max-model-len 16384 \
  --max-new-tokens 1024 --temperature 0 --seed 42

CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/llm4decompile-1.3b-v1.6 \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/llm4decompile-1.3b-v1.6-mbpp-decompile.json \
  --gpu-memory-utilization 0.20 --max-model-len 16384 \
  --max-new-tokens 1024 --temperature 0 --seed 42
```

Generate the C-only LLM4Decompile-6.7B-v1.5 row with balanced O0--O3
sharding over the available GPUs:

```bash
python evaluation/mbpp/direct_c.py --gpus 0,1
```

Fourteen valid inputs exceed the pre-registered prompt-plus-output context
budget. The runner writes an empty candidate and `context_overflow` for them,
so they remain failures in every direct-generation model's denominator.

Reproduce ReF's C-only preprocessing and tool-augmented inference:

```bash
python dataset/build_ref_decompile_eval.py \
  --dataset decompile-eval/mbpp-decompile.json \
  --output decompile-eval/mbpp-decompile-ref-c.json \
  --ref-root ../third_party/ReF-Dec --workers 16

CUDA_VISIBLE_DEVICES=1 vllm serve /home/jiang/model/ref-decompile-merged \
  --served-model-name ref-decompile --gpu-memory-utilization 0.23 \
  --dtype bfloat16 --seed 42 --max-model-len 16384 \
  --max-num-seqs 32 --enforce-eager --enable-auto-tool-choice \
  --tool-call-parser mistral --port 8011

python evaluation/run_inference_ref.py \
  --dataset decompile-eval/mbpp-decompile-ref-c.json \
  --output results/predictions/ref-decompile-6.7b-mbpp-c.json \
  --ref-eval ../third_party/ReF-Dec/eval/eval.py \
  --base-url http://127.0.0.1:8011/v1 --model-name ref-decompile \
  --tokenizer /home/jiang/model/ref-decompile-merged \
  --concurrency 32 --max-model-len 16384 --max-new-tokens 1024
```

Run SLaDe only on its released C/x86/O0,O3 scope, regenerating the GAS
representation expected by the authors' tokenizer:

```bash
CUDA_VISIBLE_DEVICES=1 python evaluation/run_inference_slade.py \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/slade-released-x86-o0-o3-mbpp-c.json \
  --o0-model ../third_party/slade_artifact/output/export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best \
  --o3-model ../third_party/slade_artifact/output/export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best \
  --assembly-source gcc-source --device cuda:0 --batch-size 16 \
  --num-beams 5 --num-return-sequences 5 --max-new-tokens 512
```

The runner emits 1,948 predictions (974 C functions at O0 and O3). O1/O2 and
C++ are absent rather than silently mapped to another checkpoint or counted as
failures.

For the separately labelled all-level diagnostic, use the O3 checkpoint
zero-shot on the optimized O1/O2 inputs:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_slade.py \
  --dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/slade-released-x86-allopts-mbpp-c.json \
  --o0-model ../third_party/slade_artifact/output/export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best \
  --o3-model ../third_party/slade_artifact/output/export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best \
  --use-o3-for-o1-o2 --assembly-source gcc-source --device cuda:0 \
  --batch-size 16 --num-beams 5 --num-return-sequences 5
```

Run the released sc²dec FAE adapter and its self-constructed-context second
pass. The adapter is attached to LLM4Decompile-6.7B-v1.5, as specified by the
authors. For MBPP, the recompiler restores `c_prefix` dependencies and selects
GCC or G++17 by language:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_sccdec.py \
  --base-model /home/jiang/model/llm4decompile-6.7b-v1.5 \
  --adapter /home/jiang/model/sccdec-lora \
  --tokenizer-source base \
  --lora-scaling standard \
  --dataset decompile-eval/mbpp-decompile-c.json \
  --output results/predictions/sccdec-6.7b-standard-lora-mbpp-c.json \
  --one-shot --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --max-new-tokens 1024 \
  --compile-workers 32 --seed 42
```

The corrected C-only run generated all 3,896 initial outputs and used
self-constructed context for 3,163. To revise only context construction while
preserving an audited first pass, add:

```bash
--initial-predictions \
  results/predictions/sccdec-6.7b-standard-lora-mbpp-c.json
```

Nova requires address-aware input and its pinned Transformers 4.40
environment. Recompile the C rows to relocatable objects with the authors'
`gcc -c`/`objdump -d` protocol and exact normalizer, then retain the authors'
20-sample decoder:

```bash
python dataset/build_nova_decompile_bench.py \
  --dataset decompile-eval/mbpp-decompile.json \
  --output decompile-eval/mbpp-decompile-nova-c.json \
  --nova-root ../third_party/nova --workers 32

CUDA_VISIBLE_DEVICES=0 .venv-nova/bin/python evaluation/run_inference_nova.py \
  --model /home/jiang/model/nova-1.3b-bcr \
  --dataset decompile-eval/mbpp-decompile-nova-c.json \
  --reference-dataset decompile-eval/mbpp-decompile.json \
  --output results/predictions/nova-1.3b-mbpp-c.json \
  --device cuda:0 --max-new-tokens 512 \
  --temperature 0.2 --top-p 0.95 --num-return-sequences 20 \
  --checkpoint-every 100 --resume --seed 42

python evaluation/evaluate_sampled_predictions.py \
  --predictions results/predictions/nova-1.3b-mbpp-c.json \
  --output results/metrics/nova-1.3b-mbpp-c-pass-at-k.json \
  --candidate-field candidates --k 1 10 \
  --compiler gcc --workers 32 --no-strict
```

The released Nova BCR checkpoint generates C, so the adapter contains all
3,896 C/O0--O3 rows and excludes C++ rather than counting it as failure. The
manifest pins the source and output hashes, compiler versions, Nova revision,
and confirms 3,896/3,896 preprocessing successes.

Score any completed prediction file with the same executable oracle:

```bash
python evaluation/evaluate_decompile_bench.py \
  --predictions results/predictions/NAME.json \
  --output results/metrics/NAME.json \
  --benchmark-metadata decompile-eval/mbpp-decompile.json \
  --reference-validation results/metrics/mbpp-decompile-reference-validation.json \
  --workers 64 --bootstrap-samples 10000 --seed 42 \
  '--crypto-library=-l:libcrypto.so.3'
```

After scoring both common-scope models, compute a paired difference interval
by resampling the 974 underlying MBPP problems:

```bash
python evaluation/compare_decompile_bench.py \
  --left results/metrics/decir-1.3b-mbpp-decompile.json \
  --right results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json \
  --left-name DecIR-1.3B --right-name LLM4Decompile-1.3B-v1.5 \
  --output results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json \
  --bootstrap-samples 10000 --seed 42
```

### Re-run external baselines

SLaDe (only the x86 O0/O3 checkpoints exist in the public archive):

```bash
python evaluation/run_inference_slade.py \
  --dataset decompile-eval/decompile-eval-executable-gcc-obj.json \
  --output results/predictions/slade-released-x86-o0-o3-gcc-compatible.json \
  --o0-model ../third_party/slade_artifact/output/export-new_train-2023-04-09-1854-b799-06dc-checkpoint_best \
  --o3-model ../third_party/slade_artifact/output/export-new_train-2023-04-28-2313-1dff-748d-checkpoint_best \
  --assembly-source gcc-source --device cuda:0 --batch-size 16 \
  --num-beams 5 --num-return-sequences 5 --max-new-tokens 512

python evaluation/apply_slade_type_inference.py \
  --predictions results/predictions/slade-released-x86-o0-o3-gcc-compatible.json \
  --output results/predictions/slade-released-x86-o0-o3-gcc-compatible-typed.json \
  --slade-root ../third_party/slade_artifact --workers 16

python evaluation/evaluate_sampled_predictions.py \
  --predictions results/predictions/slade-released-x86-o0-o3-gcc-compatible-typed.json \
  --candidate-field typed_candidates --ordered-candidates --k 1 5 \
  --output results/metrics/slade-released-x86-o0-o3-gcc-compatible-typed-beams-legacy.json \
  --compiler gcc --workers 32 --no-strict
```

Nova must run in its Transformers-4.40 environment:

```bash
.venv-nova/bin/python evaluation/run_inference_nova.py \
  --model /home/jiang/model/nova-1.3b-bcr \
  --dataset /home/jiang/model/nova-1.3b-bcr/humaneval_decompile_nova_1.3b.json \
  --reference-dataset decompile-eval/decompile-eval-executable-gcc-obj.json \
  --output results/predictions/nova-1.3b-rerun-seed42-gcc.json \
  --max-new-tokens 512 --temperature 0.2 --top-p 0.95 \
  --num-return-sequences 20 --seed 42

python evaluation/evaluate_sampled_predictions.py \
  --predictions results/predictions/nova-1.3b-rerun-seed42-gcc.json \
  --output results/metrics/nova-1.3b-rerun-seed42-gcc-legacy.json \
  --candidate-field candidates --k 1 10 --compiler gcc \
  --workers 32 --no-strict
```

sc²dec local LoRA diagnostic:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_sccdec.py \
  --base-model /home/jiang/model/llm4decompile-6.7b-v1.5 \
  --adapter /home/jiang/model/sccdec-lora \
  --tokenizer-source base \
  --lora-scaling standard \
  --dataset decompile-eval/decompile-eval-executable-gcc-obj.json \
  --output \
    results/predictions/sccdec-6.7b-standard-lora-humaneval-gcc.json \
  --one-shot --tensor-parallel-size 1 --max-new-tokens 1024 --seed 42
```

All saved prediction files use the same evaluator:

```bash
python evaluation/evaluate_predictions.py \
  --predictions results/predictions/NAME.json \
  --output results/metrics/NAME-legacy.json \
  --compiler gcc --workers 32 --no-strict \
  --bootstrap-samples 10000 --seed 42
```

## 1. Build raw, safe, and legacy IR records

```bash
export EXEBENCH_DIR=/path/to/exebench
export COMPILED_DIR=/path/to/decir/compiled

python dataset/1_get_exebench_asm_llvmir.py \
  --data_dir "$EXEBENCH_DIR" \
  --split train_real_compilable \
  --start_idx 0 \
  --num 885074 \
  --batch_size 10000 \
  --workers 8 \
  --output "$COMPILED_DIR/train-start.jsonl"
```

Each output record contains:

- `-O*_ll_raw`: compiler output;
- `-O*_ll_safe`: debug-only material removed, verifier-oriented;
- `-O*_ll`: exact legacy normalization used by released checkpoints.

To create the held-out ExeBench test records, rerun the command with the
dataset's test split (usually `test_real_compilable`) and a separate output
directory. Confirm the split name exposed by your local ExeBench checkout.

## 2. Build ablation datasets without split leakage

Set the exact tokenizer so token matching uses model tokens rather than the
fallback whitespace estimate:

```bash
export TOKENIZER=deepseek-ai/deepseek-coder-1.3b-base
export DATA_ROOT=/path/to/decir/recipes

python dataset/build_experiment_dataset.py \
  --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/safe/train/data.jsonl" \
  --recipe full --ir safe --compiler clang --partition train \
  --validation-fraction 0.01 --seed 42 --tokenizer "$TOKENIZER"

python dataset/build_experiment_dataset.py \
  --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/safe/validation/data.jsonl" \
  --recipe full --ir safe --compiler clang --partition validation \
  --validation-fraction 0.01 --seed 42 --tokenizer "$TOKENIZER"
```

Repeat the training command with these registered conditions:

```bash
# Raw IR
python dataset/build_experiment_dataset.py --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/raw/train/data.jsonl" --recipe full --ir raw \
  --compiler clang --partition train --seed 42 --tokenizer "$TOKENIZER"

# Released lossy normalization
python dataset/build_experiment_dataset.py --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/legacy/train/data.jsonl" --recipe full --ir legacy \
  --compiler clang --partition train --seed 42 --tokenizer "$TOKENIZER"

# IR-content control with a deterministic shuffled IR target
python dataset/build_experiment_dataset.py --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/shuffled/train/data.jsonl" --recipe full --ir shuffled-safe \
  --compiler clang --partition train --seed 42 --tokenizer "$TOKENIZER"

# Direct-only control repeated to match the full safe target-token budget
python dataset/build_experiment_dataset.py --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/direct-matched/train/data.jsonl" --recipe direct-repeat --ir safe \
  --compiler clang --partition train --seed 42 --tokenizer "$TOKENIZER"

# Task-1-only diagnostic model
python dataset/build_experiment_dataset.py --input "$COMPILED_DIR/train-*.jsonl" \
  --output "$DATA_ROOT/task1/train/data.jsonl" --recipe align-only --ir safe \
  --compiler clang --partition train --seed 42 --tokenizer "$TOKENIZER"
```

Create the corresponding `validation/data.jsonl` for every condition using the
same options plus `--partition validation`. The manifest next to every JSONL
records source files, split seed, function count, row count, and token budgets.

## 3. Tokenize and train three seeds

From `project/train`, prepare one recipe and run the seed sweep:

```bash
cd train

python prepare_pretrain_dataset.py \
  --data_input_dirs "$DATA_ROOT/safe/train" \
  --tokenizer_dir "$TOKENIZER" \
  --data_output_dirs "$DATA_ROOT/safe/prepared" \
  --max_length 4096 \
  --num_spliced_dataset_bins 1
```

Then train seeds 41, 42, and 43:

```bash
export PRETRAINED_MODEL="$TOKENIZER"
export TRAIN_DATA="$DATA_ROOT/safe/prepared/arrow/part-00000"
export OUTPUT_ROOT=/path/to/decir/runs
export RUN_PREFIX=decir-safe
export SEEDS="41 42 43"
bash run_seed_sweep.sh

cd ..
```

Use the same optimizer steps, batch configuration, and seeds for every
condition. `run_train.sh` writes exact settings to
`$OUTPUT_ROOT/configs/<run>.json`.

## 4. Select checkpoints without touching HumanEval

For every saved checkpoint, compute direct-task NLL on the disjoint ExeBench
validation JSONL:

```bash
python evaluation/evaluate_validation_loss.py \
  --model /path/to/checkpoint/modeling \
  --tokenizer "$TOKENIZER" \
  --validation-jsonl "$DATA_ROOT/safe/validation/data.jsonl" \
  --output results/validation/safe-seed41-checkpoint400.json

python evaluation/select_checkpoint.py \
  --results "results/validation/safe-seed41-checkpoint*.json" \
  --metric validation_nll \
  --output results/validation/safe-seed41-selected.json
```

Select the minimum-NLL checkpoint once per seed. Do not choose a checkpoint
from HumanEval curves.

## 5. Common one-shot evaluation and uncertainty

```bash
python evaluation/run_inference_vllm.py \
  --model /path/to/selected/modeling \
  --tokenizer "$TOKENIZER" \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --output results/safe-seed41-predictions.json \
  --mode direct --temperature 0 --top-p 1 --seed 42 \
  --tensor-parallel-size 1 --max-new-tokens 1024

python evaluation/evaluate_predictions.py \
  --predictions results/safe-seed41-predictions.json \
  --output results/safe-seed41-evaluation.json \
  --compiler gcc --workers 16 --bootstrap-samples 10000 --seed 42

python evaluation/aggregate_runs.py \
  --results results/safe-seed{41,42,43}-evaluation.json \
  --metric run_rate \
  --output results/safe-three-seed.json
```

Run the same evaluator on predictions released by Nova, ReF Decompile,
SALT4Decompile, SK2Decompile, and other baselines. Published numbers with a
different compiler or sampling protocol should not be inserted into the
common-protocol table.

## 6. Diagnose the Task-1-only collapse

Task 1 requires a longer generation budget. The executed ablation used
1,024, 2,048, and 4,096 new tokens; for example, the 4,096-token run is:

```bash
python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/DecIR-1.3B \
  --tokenizer /home/jiang/model/DecIR-1.3B \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --output results/predictions/decir-1.3b-task1-clang.json \
  --mode task1 --temperature 0 --seed 42 \
  --max-model-len 16384 --max-new-tokens 4096

python evaluation/analyze_task1_outputs.py \
  --predictions results/predictions/decir-1.3b-task1-clang.json \
  --output results/metrics/decir-1.3b-task1-clang-analysis.json \
  --clang clang --c-compiler gcc
```

The analysis separates exact/fuzzy/missing markers, length truncation, source
presence, IR validity (`clang -x ir`), C compilation, and fixed-test
re-executability.

## 7. Stronger semantic and structural checks

For the scalar-signature subset:

```bash
python evaluation/differential_test.py \
  --predictions results/safe-seed42-predictions.json \
  --output results/safe-seed42-differential.json \
  --compiler clang --trials 1000 --seed 42
```

The output includes eligibility coverage; do not present the eligible-subset
pass rate without it.

For legacy reproduction only, build the held-out ExeBench O3 structural
benchmark from a local Parquet partition, generate predictions with
`run_inference_vllm.py`, and report contextual compilation and structure.
Use `evaluation/exebench/` above for the confirmatory executable comparison:

```bash
python dataset/build_exebench_eval.py \
  --parquet /home/jiang/share/dataset/exebench_parquet/test_real/part-00000.parquet \
  --split test_real \
  --output decompile-eval/exebench-test-real-gcc-o3.json \
  --compiler gcc --optimization O3 --limit 5000

CUDA_VISIBLE_DEVICES=0 python evaluation/run_inference_vllm.py \
  --model /home/jiang/model/DecIR-1.3B \
  --dataset decompile-eval/exebench-test-real-gcc-o3.json \
  --output results/predictions/decir-1.3b-exebench-test-real-gcc-o3.json \
  --mode direct --max-model-len 16384 --max-new-tokens 1024 \
  --temperature 0 --top-p 1 --seed 42

python evaluation/evaluate_predictions.py \
  --predictions results/predictions/decir-1.3b-exebench-test-real-gcc-o3.json \
  --output results/metrics/decir-1.3b-exebench-test-real-gcc-o3-legacy.json \
  --compiler gcc --workers 32 --no-strict

python evaluation/evaluate_structure.py \
  --predictions results/predictions/decir-1.3b-exebench-test-real-gcc-o3.json \
  --output results/metrics/decir-1.3b-exebench-test-real-gcc-o3-structure.json
```

## 8. Frontier-model feedback baseline

The script accepts any OpenAI-compatible `/chat/completions` endpoint. It does
not reveal benchmark assertion text.

```bash
export DECIR_API_KEY='...'

python evaluation/run_agent_baseline.py \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --output results/frontier-compile-feedback.json \
  --api-url https://provider.example/v1/chat/completions \
  --model MODEL_ID \
  --rounds 3 --feedback compile --temperature 0

python evaluation/run_agent_baseline.py \
  --dataset decompile-eval/decompile-eval-executable-clang-obj.json \
  --output results/frontier-test-feedback.json \
  --api-url https://provider.example/v1/chat/completions \
  --model MODEL_ID \
  --rounds 3 --feedback tests --temperature 0
```

Report model identifier/date, rounds, feedback mode, token usage if returned by
the provider, and wall-clock/API cost. Test-feedback results are a separate
tool-augmented setting and are not a one-shot comparison.

## 9. Blinded readability study

Prepare balanced, method-blinded packets only from task keys meeting the
pre-registered semantic filter (for example, all compared outputs pass the
fixed tests). Keep the key file hidden from raters:

```bash
python evaluation/select_common_passes.py \
  --evaluation DecIR=results/safe-seed42-evaluation.json \
  --evaluation Baseline=results/baseline-evaluation.json \
  --metric re_executable \
  --output results/readability-eligible-task-keys.json

python evaluation/prepare_readability_study.py \
  --predictions DecIR=results/safe-seed42-predictions.json \
  --predictions Baseline=results/baseline-predictions.json \
  --eligible-task-ids results/readability-eligible-task-keys.json \
  --sample-size 50 --seed 42 \
  --output-csv results/readability/rater-template.csv \
  --key-output results/readability/private-key.json

python evaluation/analyze_readability_study.py \
  --ratings results/readability/rater1.csv \
            results/readability/rater2.csv \
            results/readability/rater3.csv \
  --key results/readability/private-key.json \
  --output results/readability/summary.json \
  --bootstrap-samples 10000 --seed 42
```

Obtain any ethics/participant approval required by the target venue before
recruiting raters. Report expertise, number of raters, sampling/filtering,
instructions, compensation, rating scale, and intervals.

## 10. Populate LaTeX without manual transcription

Executed result export:

```bash
python evaluation/export_latex.py --output ../paper/generated_results.tex \
  --value TaskOneMarker=results/metrics/decir-1.3b-task1-clang-analysis.json:summary.marker_exact.rate \
  --value TaskOneTruncated=results/metrics/decir-1.3b-task1-clang-analysis.json:summary.truncated.rate \
  --value TaskOneIRValid=results/metrics/decir-1.3b-task1-clang-analysis.json:summary.ir_valid.rate \
  --value TaskOneCompile=results/metrics/decir-1.3b-task1-clang-analysis.json:summary.compilable.rate \
  --value TaskOneRE=results/metrics/decir-1.3b-task1-clang-analysis.json:summary.re_executable.rate \
  --value DifferentialCoverage=results/metrics/decir-1.3b-clang-differential.json:summary.coverage \
  --value DifferentialPass=results/metrics/decir-1.3b-clang-differential.json:summary.pass_rate_among_eligible \
  --value ExeBenchCompile=results/metrics/decir-1.3b-exebench-test-real-gcc-o3-legacy.json:summary.mean.compile_rate \
  --value ExeBenchEdit=results/metrics/decir-1.3b-exebench-test-real-gcc-o3-structure.json:summary.token_edit_similarity \
  --value ExeBenchControl=results/metrics/decir-1.3b-exebench-test-real-gcc-o3-structure.json:summary.control_f1 \
  --value ExeBenchOperator=results/metrics/decir-1.3b-exebench-test-real-gcc-o3-structure.json:summary.operator_f1 \
  --value ExeBenchSignature=results/metrics/decir-1.3b-exebench-test-real-gcc-o3-structure.json:summary.signature_exact \
  --value MBPPDecIRCompile=results/metrics/decir-1.3b-mbpp-decompile.json:summary.overall.compile_rate \
  --value MBPPDecIRRun=results/metrics/decir-1.3b-mbpp-decompile.json:summary.overall.run_rate \
  --value MBPPDecIREdit=results/metrics/decir-1.3b-mbpp-decompile.json:summary.overall.edit_similarity \
  --value MBPPDecIROZero=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_optimization.O0.run_rate \
  --value MBPPDecIROOne=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_optimization.O1.run_rate \
  --value MBPPDecIROTwo=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_optimization.O2.run_rate \
  --value MBPPDecIROThree=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_optimization.O3.run_rate \
  --value MBPPDecIRCCompile=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_language.c.compile_rate \
  --value MBPPDecIRCRun=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_language.c.run_rate \
  --value MBPPDecIRCEdit=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_language.c.edit_similarity \
  --value MBPPDecIRCPPRun=results/metrics/decir-1.3b-mbpp-decompile.json:summary.by_language.cpp.run_rate \
  --value MBPPBaseCompile=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.overall.compile_rate \
  --value MBPPBaseRun=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.overall.run_rate \
  --value MBPPBaseEdit=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.overall.edit_similarity \
  --value MBPPBaseOZero=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_optimization.O0.run_rate \
  --value MBPPBaseOOne=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_optimization.O1.run_rate \
  --value MBPPBaseOTwo=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_optimization.O2.run_rate \
  --value MBPPBaseOThree=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_optimization.O3.run_rate \
  --value MBPPBaseCCompile=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_language.c.compile_rate \
  --value MBPPBaseCRun=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_language.c.run_rate \
  --value MBPPBaseCEdit=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_language.c.edit_similarity \
  --value MBPPBaseCPPRun=results/metrics/llm4decompile-1.3b-v1.5-mbpp-decompile.json:summary.by_language.cpp.run_rate \
  --value MBPPDCBenchCompile=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.overall.compile_rate \
  --value MBPPDCBenchRun=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.overall.run_rate \
  --value MBPPDCBenchEdit=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.overall.edit_similarity \
  --value MBPPDCBenchOZero=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_optimization.O0.run_rate \
  --value MBPPDCBenchOOne=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_optimization.O1.run_rate \
  --value MBPPDCBenchOTwo=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_optimization.O2.run_rate \
  --value MBPPDCBenchOThree=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_optimization.O3.run_rate \
  --value MBPPDCBenchCCompile=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_language.c.compile_rate \
  --value MBPPDCBenchCRun=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_language.c.run_rate \
  --value MBPPDCBenchCEdit=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_language.c.edit_similarity \
  --value MBPPDCBenchCPPRun=results/metrics/llm4decompile-1.3b-v1.6-mbpp-decompile.json:summary.by_language.cpp.run_rate \
  --value MBPPReFCompile=results/metrics/ref-decompile-6.7b-mbpp-c.json:summary.overall.compile_rate \
  --value MBPPReFRun=results/metrics/ref-decompile-6.7b-mbpp-c.json:summary.overall.run_rate \
  --value MBPPReFEdit=results/metrics/ref-decompile-6.7b-mbpp-c.json:summary.overall.edit_similarity \
  --value MBPPSCCCompile=results/metrics/sccdec-6.7b-standard-lora-mbpp-c-final.json:summary.overall.compile_rate \
  --value MBPPSCCRun=results/metrics/sccdec-6.7b-standard-lora-mbpp-c-final.json:summary.overall.run_rate \
  --value MBPPSCCEdit=results/metrics/sccdec-6.7b-standard-lora-mbpp-c-final.json:summary.overall.edit_similarity \
  --value MBPPSCCInitialRun=results/metrics/sccdec-6.7b-standard-lora-mbpp-c-initial.json:summary.overall.run_rate \
  --value MBPPSCCDelta=results/metrics/sccdec-6.7b-standard-lora-mbpp-final-vs-initial-paired.json:by_language.c.differences_percentage_points.re_executable.left_minus_right \
  --value MBPPSCCDeltaLow=results/metrics/sccdec-6.7b-standard-lora-mbpp-final-vs-initial-paired.json:by_language.c.differences_percentage_points.re_executable.clustered_ci95.0 \
  --value MBPPSCCDeltaHigh=results/metrics/sccdec-6.7b-standard-lora-mbpp-final-vs-initial-paired.json:by_language.c.differences_percentage_points.re_executable.clustered_ci95.1 \
  --value MBPPNovaCompile=results/metrics/nova-1.3b-mbpp-c-first-sample.json:summary.overall.compile_rate \
  --value MBPPNovaRun=results/metrics/nova-1.3b-mbpp-c-first-sample.json:summary.overall.run_rate \
  --value MBPPNovaEdit=results/metrics/nova-1.3b-mbpp-c-first-sample.json:summary.overall.edit_similarity \
  --value MBPPNovaPassOne=results/metrics/nova-1.3b-mbpp-c-pass-at-k.json:re_executable.mean.pass@1 \
  --value MBPPNovaPassTen=results/metrics/nova-1.3b-mbpp-c-pass-at-k.json:re_executable.mean.pass@10 \
  --value MBPPNovaStandaloneCompile=results/metrics/nova-1.3b-mbpp-c-pass-at-k.json:recompilable.mean.pass@1 \
  --value MBPPSLaDeCompile=results/metrics/slade-released-x86-o0-o3-mbpp-c.json:summary.overall.compile_rate \
  --value MBPPSLaDeRun=results/metrics/slade-released-x86-o0-o3-mbpp-c.json:summary.overall.run_rate \
  --value MBPPSLaDeEdit=results/metrics/slade-released-x86-o0-o3-mbpp-c.json:summary.overall.edit_similarity \
  --value MBPPSLaDeOverflow=results/metrics/slade-released-x86-o0-o3-mbpp-c.json:generation_audit.context_overflow \
  --value MBPPPrimaryDifference=results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json:overall.differences_percentage_points.re_executable.left_minus_right \
  --value MBPPPrimaryDifferenceLow=results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json:overall.differences_percentage_points.re_executable.clustered_ci95.0 \
  --value MBPPPrimaryDifferenceHigh=results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json:overall.differences_percentage_points.re_executable.clustered_ci95.1 \
  --value MBPPPrimaryCDifference=results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json:by_language.c.differences_percentage_points.re_executable.left_minus_right \
  --value MBPPPrimaryCPPDifference=results/metrics/decir-vs-llm4decompile-v1.5-mbpp-paired.json:by_language.cpp.differences_percentage_points.re_executable.left_minus_right \
  --value MBPPStrongDifference=results/metrics/llm4decompile-v1.6-vs-decir-mbpp-paired.json:overall.differences_percentage_points.re_executable.left_minus_right \
  --value MBPPStrongDifferenceLow=results/metrics/llm4decompile-v1.6-vs-decir-mbpp-paired.json:overall.differences_percentage_points.re_executable.clustered_ci95.0 \
  --value MBPPStrongDifferenceHigh=results/metrics/llm4decompile-v1.6-vs-decir-mbpp-paired.json:overall.differences_percentage_points.re_executable.clustered_ci95.1
```

Add future controlled-training macros to the same exporter call: the generated
file is replaced on each invocation.
