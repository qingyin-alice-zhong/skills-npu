# NPU Kernel Auto-Optimization Agent

## Overview
Iteratively generate and optimize vectorized NPU kernels for all kernels in `llm_codegen_auto/npueval_dataset`, up to 5 iterations each. Measure speedup vs. scalar baseline after each passing iteration and improve.

Commands and file writes will be auto-approved in this session.

---

## Setup (once)
1. Working directory: `/home/qz425`
2. Working env:`source /home/qz425/env_restore_legacy.sh`
3. Read `llm_codegen_auto/npueval_dataset`
   *(test.py imports `analyze_trace` from its parent dir at runtime)*
4. Determine output folder: scan for existing `llm_codegen_auto/output_N/`, set `x = max(N) + 1` (or 1 if none). Create empty `llm_codegen_auto/output_x/{kernel_name}/` for every kernel in npueval_dataset.

---

## kernel_func.cc Generation Reference

Use skill `llm_codegen_auto/.claude/skills/npu-kernel-gen` as the authoritative reference for writing vectorized `kernel_func.cc` files — it specifies available AIE intrinsics, vector types, and constraints. Key conventions:
- Vectorize with `aie` intrinsics (e.g. `aie::abs`, `aie::add`, etc.), `vec_factor=32`
- Refer to `canonical_scalar_allo.cc` for the operation semantics to preserve

Read and internalize this skill `llm_codegen_auto/.claude/skills/npu-kernel-gen`  before generating the first kernel, then keep it as reference throughout.

---

## Per-Kernel Loop (repeat for each kernel, 5 iterations max)

### Step 0 — Measure scalar baseline (once per kernel)
```
python llm_codegen_auto/npueval_dataset/{kernel}/test.py --kernel_path llm_codegen_auto/npueval_dataset/{kernel}/canonical_scalar_allo.cc
```
Parse output for: `First/Min/Avg/Max cycles is A/ B/ C/ D` → `scalar_cycles = C` (Avg field).

### Step 1 — Generate kernel (iter_n, starting at n=1)
Reference `canonical_scalar_allo.cc` for operation semantics. Apply vectorization per the npu-kernel-gen skill. Write output to:
```
llm_codegen_auto/output_x/{kernel}/iter_n/kernel_func.cc
```
Note: Generate and test one iter at a time — do not pre-generate multiple iterations.
Only start iter_{n+1} after reviewing iter_n's kernel and output_detail.md; use it as the direct reference for the next optimization.


### Step 2 — Compile & test
```
python llm_codegen_auto/npueval_dataset/{kernel}/test.py --kernel_path llm_codegen_auto/output_x/{kernel}/iter_n/kernel_func.cc
```

### Step 3 — Parse results and write output files
Save to `llm_codegen_auto/output_x/{kernel}/iter_n/`:

- `result.md` — brief summary: compile status, pass/fail, scalar_cycles, kernel_cycles, speedup
- `output_detail.md` — full AIE compile & verification output verbatim. **Read this before writing the next iteration's kernel_func.cc.**

Parse:
- Correctness: look for `PASS!` or `FAIL!`
- Performance: `First/Min/Avg/Max cycles is A/ B/ C/ D` → `kernel_cycles = C`
- **`speedup = scalar_cycles / kernel_cycles`**

### Step 4 — Decide next action

| Outcome | Action |
|---------|--------|
| Compile error | Fix error based on `output_detail.md`, write `iter_{n+1}/kernel_func.cc`, go to Step 2 |
| FAIL (wrong results) | Analyze assertion diff in `output_detail.md`, fix correctness, write `iter_{n+1}/kernel_func.cc`, go to Step 2 |
| PASS, iterations remain | Attempt further vectorization to improve speedup, write `iter_{n+1}/kernel_func.cc`, go to Step 2 |
| PASS, n=5 | Select best passing iter (highest speedup). If all iters fail, record "no passing iteration". Write `summary.md`, move to next kernel |

If `best_speedup < 1.0`, still record it but annotate as "no improvement over scalar".

### Step 5 — Per-kernel summary
Write `llm_codegen_auto/output_x/{kernel}/summary.md`:

| iter | compile | pass/fail | kernel_cycles | speedup |
|------|---------|-----------|---------------|---------|
| 1    | ...     | ...       | ...           | ...     |
| ...  |         |           |               |         |
| best | (highlight best passing iter) |

---

## Final Output
After all kernels are done, write `llm_codegen_auto/output_x/README.md` — summary table:

| kernel | best_iter | best_speedup | note |
|--------|-----------|--------------|------|
| ...    | ...       | ...          | ...  |

---

注意：请用中文对 agent 窗口面向的用户进行解释和汇报进度，但所有写入文件的内容（result.md、summary.md、README.md 等）用英文。
