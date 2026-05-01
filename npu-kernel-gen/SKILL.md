---
name: npu-kernel-gen
description: Generate AMD XDNA NPU kernel .cc files and allo test.py files for AIE operations. Use when the user asks to create a new NPU kernel, AIE kernel, wants to add a new operation to the npueval_dataset, or wants to convert a PyTorch operation to an NPU kernel. Handles both small single-tile kernels and large multi-tile kernels with tiling/mapping.
argument-hint: "<operation_name> <dtype> [buffer_sizes...] or <pytorch_op_description>"
---

# NPU Kernel & Test Generator

Generate NPU kernel `.cc` files and `test.py` for AIE operations. Supports two modes:
- **Small kernel** (data fits in single tile, <=1024 elements per buffer): generates `kernel_func.cc` (vectorized) + `test.py`
- **Large kernel** (data exceeds single tile): generates `{name}.cc` (tiled kernel) + `{name}_test.py` (with tiling/mapping logic)

## Step 0: Determine Kernel Size Class

Given the user's operation (possibly a PyTorch `nn.Module` or functional op), calculate total buffer sizes:

```
total_input_elements  = product of input tensor dimensions
total_output_elements = product of output tensor dimensions
total_param_elements  = weights + bias (if any)
```

**Classification:**
- If ALL buffers (input, output, param) each have **<= 1024 elements** AND total per-buffer **<= 4KB** (considering dtype size) → **Small kernel** (single tile)
- Otherwise → **Large kernel** (needs tiling + multi-core mapping)

### NPU Hardware Constraints (CRITICAL)

These constraints MUST be respected in ALL generated code:

1. **DMA size limit**: each dimension of `dma_memcpy_nd` size field is 10-bit → max 1023 per dimension. Flatten multi-dimensional data to 1D arrays.
2. **Local tile memory**: each AIE tile has ~64KB local data memory. All input + output + param buffers for ONE tile invocation must fit within this.
3. **Buffer size rule of thumb**: keep each single buffer ≤ 4KB for safety (leaving room for stack, code, etc.)
4. **Data type sizes**: int8=1B, int16/bfloat16=2B, int32/float32=4B
5. **Maximum practical buffer**: ~1024 elements for 4-byte types (4KB), ~2048 for 2-byte types, ~4096 for 1-byte types

---

## Mode A: Small Kernel Generation

### Inputs from User

Parse `$ARGUMENTS` to extract:
- **operation_name**: e.g., `relu`, `sigmoid`, `matmul_16x16`, `conv1d`, `avgpool2d`
- **dtype**: e.g., `int8`, `int16`, `int32`, `bfloat16`, `float32`
- **buffer_sizes** (optional): input/output buffer sizes. If not given, default to 1024 for inputs.

The full kernel name is `{operation_name}_{dtype}` (e.g., `relu_int8`, `sigmoid_bfloat16`).

### Directory Structure

Create directory: `llm_codegen_spec/npueval_dataset/{operation_name}_{dtype}/`

```
{operation_name}_{dtype}/
├── kernel_func.cc              # Primary AIE kernel (vectorized-first)
└── test.py                     # Allo dataflow test harness
```

### File Templates

Do NOT fill in abstract templates. See Step 2 — start from the bundled references under `${CLAUDE_SKILL_DIR}/references/` and adapt minimally.

---

## Mode B: Large Kernel Generation (Tiling + Multi-Core Mapping)

When a PyTorch operation's tensors exceed single-tile capacity, the kernel must be **decomposed into tiles**.

### Inputs from User

The user provides either:
- A PyTorch `nn.Module` class / functional op description
- Or explicit: operation name, full tensor shapes, dtype

### Key Principle: The .cc Kernel Operates on ONE TILE

The `.cc` file implements the operation on **one tile only**. The `test.py` handles:
1. Decomposing full tensors into tiles
2. Dispatching tiles to multiple AIE cores via `mapping=[N]`
3. Reassembling tiled outputs into the full result
4. Comparing against PyTorch reference

### Tiling Strategy (CRITICAL)

#### Step 1: Determine Tile Dimensions

For each operation type, choose tile sizes that:
- Keep each buffer within ~4KB (1024 float32 elements)
- Align with the operation's computation pattern
- Include necessary overlap (e.g., padding/halo for convolutions)

**Common tiling patterns:**

| Operation | Tile Strategy | Example |
|-----------|--------------|---------|
| Element-wise (relu, sigmoid) | Tile along flattened dimension | tile_size=1024 |
| MatMul [M,K]x[K,N] | Tile M and N dimensions | tile_M=16, tile_N=16, full K |
| Conv2d | Tile output channels + spatial | tile_oc=8, tile_oh=8, tile_ow=8 |
| Pooling | Tile spatial dimensions | tile_h=32, tile_w=32 |
| Normalization | Tile batch/sequence, keep feature dim | tile_seq=4, full feature |

#### Step 2: Calculate Per-Tile Buffer Sizes

For convolution example:
```
TILE_INPUT_HEIGHT = TILE_OUT_HEIGHT + KERNEL_H - 1   # include receptive field overlap
TILE_INPUT_WIDTH  = TILE_OUT_WIDTH  + KERNEL_W - 1
INPUT_SIZE  = IN_CHANNELS * TILE_INPUT_HEIGHT * TILE_INPUT_WIDTH
OUTPUT_SIZE = TILE_OUT_CHANNELS * TILE_OUT_HEIGHT * TILE_OUT_WIDTH
PARAM_SIZE  = TILE_OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W + TILE_OUT_CHANNELS  # weights + bias
```

**Verify**: `INPUT_SIZE * dtype_bytes + OUTPUT_SIZE * dtype_bytes + PARAM_SIZE * dtype_bytes < 64KB`

#### Step 3: Memory Budget Check

```python
def check_tile_memory(input_size, output_size, param_size, dtype_bytes=4):
    total = (input_size + output_size + param_size) * dtype_bytes
    assert total < 65536, f"Tile memory {total}B exceeds 64KB! Reduce tile sizes."
    print(f"Tile memory: {total}B ({total/1024:.1f}KB) — OK")
```

### Directory Structure (Large Kernel)

```
decomposition_kernel/kernel_agent/kernel/{kernel_name}/
├── {kernel_name}.cc           # Tiled kernel (operates on one tile)
├── {kernel_name}_test.py      # Full tiling + mapping + verification
└── {kernel_name}.py           # PyTorch reference (read-only, from user)
```

### Templates (Large Kernel)

Do NOT fill in abstract templates. See Step 2 — start from `${CLAUDE_SKILL_DIR}/references/verified_large_kernel/conv2d_3x64_b1a_fp32.cc` and `conv2d_3x64_b1a_fp32_test.py`, then adapt minimally.

---

## Large Kernel: Tiling Rules & Gotchas

### 1. Convolution Tiling

- **Input tile must include halo/overlap**: `tile_input_h = tile_out_h + kernel_h - 1`
- **Padding**: pad the full input BEFORE extracting tiles. Do NOT pad inside the tile kernel.
- **Weight slicing**: for tiled output channels, slice `weight[oc_start:oc_start+tile_oc]`
- **Boundary tiles**: last tile may be smaller. Zero-pad the tile buffer to full tile size, but only use `actual_extent` elements from the output.
- **Param packing**: concatenate `[weight_flat, bias_flat]` into a single 1D param buffer.

### 2. MatMul Tiling

- Tile M and N (output dimensions), keep K full or tile K with accumulation.
- If tiling K: need reduction across tiles (accumulate partial sums).
- Watch for int overflow: use int32 accumulator for int8 matmul, clamp result.

### 3. Pooling Tiling

- Input tile larger than output tile by pool window size.
- Stride affects tile overlap calculation.

### 4. Element-wise / Activation Tiling

- Simplest case: just chunk the flat array into tiles of <=1024 elements.
- No overlap needed.

### 5. Priority Order for Implementation

When implementing a large kernel, follow this priority:
1. **Get it to compile** — correct buffer sizes, valid C++ syntax, correct AIE includes
2. **Get it to execute** — no segfaults, no NaN, no buffer overflows
3. **Get correct results** — match PyTorch reference within tolerance
4. **Scale up** — increase tile size or core count towards full problem size

### 6. Common Failure Modes & Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Buffer overflow / segfault | Tile buffer too large for local memory | Reduce tile dimensions |
| NaN in output | Uninitialized memory, division by zero | Zero-init all buffers, check edge cases |
| Wrong results (large error) | Index calculation mismatch between .cc and test.py | Verify index formulas match exactly |
| Wrong results (small error) | Float precision, accumulation order | Use `double` accumulator in .cc, increase tolerance |
| Compilation error | Wrong types, missing extern "C" | Check type mapping table, ensure extern "C" wraps function |
| Tiles work but full assembly wrong | Tile overlap/boundary handling bug | Check halo calculation, verify boundary tile extraction |
| Some tiles correct, others wrong | Lane/pid mapping mismatch | Check `meta_if`/`meta_elif` pid routing matches lane assignment |

### 7. float32 Precision Best Practice

For float32 kernels (especially multi-accumulation like conv/matmul):
- Use `double acc` in the .cc kernel for intermediate accumulation
- Cast back to float only at final store: `output[idx] = static_cast<float>(acc)`
- Use small random input scale (e.g., `* 0.05`) in test to reduce error amplification
- Set tolerance to `rtol=1e-2, atol=1e-2`

---

## Type Mapping Rules

### C++ Types (in .cc files)

| dtype      | kernel_func.cc |
|------------|----------------|
| int8       | `std::int8_t`  |
| int16      | `std::int16_t` |
| int32      | `std::int32_t` |
| bfloat16   | `bfloat16`     |
| float32    | `float`        |

- `kernel_func.cc`: prefer fixed-size array parameters for small kernels (e.g., `std::int8_t in[1024]`)
- Loop variables in AIE kernels use `std::uint32_t` / `std::int32_t` as needed

### Python Types (in test.py)

| dtype    | allo.ir.types import | numpy dtype      | extra import                              |
|----------|---------------------|------------------|-------------------------------------------|
| int8     | `int8`              | `np.int8`        | —                                         |
| int16    | `int16`             | `np.int16`       | —                                         |
| int32    | `int32`             | `np.int32`       | —                                         |
| bfloat16 | `bfloat16`          | `np_bfloat16`    | `from ml_dtypes import bfloat16 as np_bfloat16` |
| float32  | `float32`           | `np.float32`     | —                                         |

### Tolerance by dtype

| dtype      | rtol | atol |
|------------|------|------|
| int8/16/32 | 1e-2 | 1e-2 |
| bfloat16   | 3e-2 | 3e-2 |
| float32    | 1e-2 | 1e-2 |

### bfloat16 Special Handling

When dtype is `bfloat16`:
- Add `from ml_dtypes import bfloat16 as np_bfloat16` to test.py
- Use `np_bfloat16` as numpy dtype (not `np.bfloat16`)
- Convert to float32 before `assert_allclose`: `output.astype(np.float32)`, `ref.astype(np.float32)`

### Input Generation by Operation Type

| Category | Random generation pattern |
|----------|--------------------------|
| int8 element-wise | `np.random.randint(-100, 100, (N,), dtype=np.int8)` |
| int32 element-wise | `np.random.randint(-1000, 1000, (N,), dtype=np.int32)` |
| bfloat16/float32 | `(np.random.randn(N) * scale).astype(dtype)` |
| matmul int8 | `np.random.randint(-10, 10, ...)` (small range, avoid overflow) |
| activation (sigmoid/tanh) | `np.random.uniform(-5.0, 5.0, ...)` |
| large kernel float32 | `rng.standard_normal(...) * 0.05` (small scale for precision) |

---

## Key Patterns to Follow

### Small Kernels
1. **ExternalModule**: `input_idx` lists indices of input params, `output_idx` lists output param indices (0-based, matching C function signature order)
2. **df.kernel mapping**: `mapping=[1]` for single-tile
3. **Layout**: `Ly = Layout("R")` with `@ Ly` on all kernel params
4. **Reference function**: `Annotated[np.ndarray, "shape: (N,)"]` type hints
5. **Reference function** marked between `# Reference code starts` / `# Reference code ends`

### Large Kernels
1. **mapping=[MAPPING_CORES]** (typically 4) for multi-core parallelism
2. **pid-based routing**: `df.get_pid()` + `allo.meta_if` to route each core to its own buffers
3. **Layout**: `LyRep = [R]` (Replicate) — each core gets its own copy
4. **Per-core explicit buffers**: top-level region has N copies of each buffer (A0, A1, ..., AN-1)
5. **Tile task loop**: iterate over all tile coordinates, group into batches of MAPPING_CORES
6. **Boundary handling**: last tile may be partial — zero-pad buffer, use actual_extent for output
7. **PyTorch reference**: use `torch.nn.functional` or `nn.Module` for golden reference
8. **Warmup call**: do one dummy `mod(...)` call with zero buffers before the real tiling loop
9. **Param packing**: concatenate all weight/bias into single 1D `param` array per tile

---

## Vectorization-First Policy (Default)

For AIE/NPU kernels, **default to vectorized implementation**.
Do not generate `canonical_scalar.cc` or `canonical_scalar_allo.cc` in normal workflow.
If an operation cannot be vectorized with available AIE API, explicitly report that limitation and still keep `kernel_func.cc` as the only kernel artifact.

When vectorizing, follow these hard rules:
1. Use `#include <aie_api/aie.hpp>` and AIE vector types (`aie::vector`, `aie::accum`) where applicable.
2. Compute vector factor from dtype and kernel family conventions (e.g., `256 / (sizeof(T) * 8)` or existing reference's fixed factor).
3. Use `__restrict` pointers on hot input/output buffers.
4. Use `aie::load_v<...>` / vector arithmetic / `aie::store_v(...)` in the innermost loop.
5. Add loop scheduling pragmas for hot loops:
    - `AIE_PREPARE_FOR_PIPELINING`
    - `AIE_LOOP_MIN_ITERATION_COUNT(16)` (or a justified lower bound)
6. Keep tail handling for non-multiple vector lengths (masked/tail loop).
7. Keep `event0()` / `event1()` around the compute region for trace analysis.

Vectorization guidance should be aligned with:
- `https://github.com/Xilinx/mlir-aie/tree/main/programming_guide`
- `programming_guide/section-4/section-4c/README.md`
- `programming_guide/quick_reference.md`

---

## Workflow (MUST follow this order)

### Step 1: Parse & Classify

1. **Parse** the operation semantics from user input (PyTorch op, shapes, dtype)
2. **Classify** as small or large kernel based on buffer sizes (see Step 0 above)

### Step 2: Read Reference Files (MANDATORY before generating any code)

**You MUST read real reference files from the codebase before writing any .cc or test.py.** Do NOT rely solely on templates in this skill — always ground your output in actual working code from the repository.

Use this source priority (skill-local mirrors only):
1. `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/*` and `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/*` (primary)
2. `${CLAUDE_SKILL_DIR}/references/allo_examples/*` (secondary)
3. `${CLAUDE_SKILL_DIR}/references/verified_large_kernel/*` (task-specific)

Choose the most similar existing kernel based on operation type:

#### For Small Kernels — read from `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/` and `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/`

Select the closest match by operation category, then READ all files from these bundled references:

| Your operation type | Reference to read | Why |
|--------------------|-------------------|-----|
| Element-wise unary / binary | `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/gelu.cc` | Vectorized math kernel style |
| Normalization family | `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/norm.cc` + `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_norm.py` | ExternalModule + vectorized kernel pairing |
| MatMul / GEMM | `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/gemm.py` + `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_mapping_gemm.py` | Mapping and end-to-end build pattern |
| General AIE kernel style | `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/layer_norm.cc`, `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/softmax_bf16.cc` | Vectorized loops, accumulators, numerics |

If the bundled samples don't cover your exact case:
1. Search `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/` and `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/` for the nearest operation family.
2. Then adapt from `${CLAUDE_SKILL_DIR}/references/verified_large_kernel/` patterns for buffer packing and tiling.

#### For Large Kernels — read from skill-local GEMM/norm mirrors first, then large-kernel mirrors

**Always read these files first:**
```
Read: ${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/gemm.py
Read: ${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/norm.cc
Read: ${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_mapping_gemm.py
```

**Then read these bundled mirrors when needed:**
```
Read: ${CLAUDE_SKILL_DIR}/references/verified_large_kernel/conv2d_3x64_b1a_fp32.cc
Read: ${CLAUDE_SKILL_DIR}/references/verified_large_kernel/conv2d_3x64_b1a_fp32_test.py
Read: ${CLAUDE_SKILL_DIR}/references/verified_large_kernel/conv2d_3x64_b1a_fp32.py
```

**Also read mapping/layout references from bundled `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/`** (pick the most relevant):

| Your operation type | Additional reference to read |
|--------------------|------------------------------|
| Conv / spatial ops | `allo_examples/allo_tests/test_norm.py` + `allo_examples/allo_kernels/norm.cc` |
| MatMul / GEMM | `allo_examples/allo_tests/test_mapping_gemm.py` |
| General mapping patterns | `allo_examples/allo_tests/test_mapping_basic.py` |
| Multi-core parallel | `allo_examples/allo_tests/test_collective_communication.py` |
| Meta-programming (meta_if/meta_for) | `allo_examples/allo_tests/test_meta_for.py` |

#### For ANY kernel — additionally read vector programming guidance:
- `https://github.com/Xilinx/mlir-aie/tree/main/programming_guide`
- `programming_guide/section-4/section-4c/README.md`
- `programming_guide/quick_reference.md`

#### For ANY kernel — optionally read from bundled references:
- `${CLAUDE_SKILL_DIR}/references/api_doc/api_doc.md` — AIE vector/accumulator API details
- `${CLAUDE_SKILL_DIR}/references/allo_docs/dataflow.rst` — Allo dataflow concept and patterns
- `${CLAUDE_SKILL_DIR}/references/allo_docs/memory.py` — Layout/Shard/Replicate API
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/gelu.cc` — LUT-based activation kernel example
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/layer_norm.cc` — normalization kernel example
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/softmax_bf16.cc` — softmax bfloat16 example

### Step 3: Plan Tile Geometry (large kernels only)

1. **Calculate** tile dimensions respecting hardware constraints
2. **Verify memory budget**: `(INPUT_SIZE + OUTPUT_SIZE + PARAM_SIZE) * dtype_bytes < 64KB`
3. **Report** tile geometry to user before generating code

### Step 4: Generate Code

Generate files by **adapting the reference code you just read**, NOT by filling in abstract templates:

1. **Start from the reference test.py** you read in Step 2
2. **Modify** the reference to match the new operation's semantics:
    - Keep vectorized path as the default implementation (use `aie::load_v`/`aie::store_v` and loop pragmas)
    - Do not emit scalar-only kernel bodies unless vectorization is explicitly impossible
   - Change kernel name, function signature, buffer sizes
   - Rewrite the reference function for the new operation
   - Adjust input generation (dtype, range, shape)
   - Update ExternalModule `input_idx` / `output_idx`
   - For large kernels: adjust tile dimensions, tiling loops, tile extraction logic
3. **Keep unchanged** everything that is boilerplate:
   - Import structure, utils import, `sys.path` setup
   - Layout declarations, `analyze_trace` call
   - argparse `__main__` block
   - `MLIR_AIE_INSTALL_DIR` check pattern
   - For large kernels: `iter_tile_starts`, `df.region`/`df.kernel` structure, pid routing pattern

Similarly for .cc files:
1. **Start from the reference .cc** you read
2. **Modify** the computation logic for the new operation
3. **Keep unchanged** the AIE boilerplate (copyright, includes, extern "C", event0/event1)

### Step 5: Cross-Check Consistency

Before finalizing, verify these match between .cc and test.py:
- [ ] Buffer sizes (array dimensions in .cc == SIZE constants in test.py)
- [ ] Function name in .cc == `top=` string in ExternalModule
- [ ] Number and order of parameters
- [ ] `input_idx` / `output_idx` match the C function signature order
- [ ] dtype in .cc matches allo type and numpy dtype in test.py
- [ ] For large kernels: tile constants are consistent everywhere

### Step 6: Report to User

- List generated files
- For large kernels: tile geometry, memory budget, estimated tile/group count
- Suggest test command:
    - Small kernel: `python test.py --kernel_path kernel_func.cc`
  - Large kernel: `python main.py --kernel {name}`

## Bundled Reference Files Index

All references are bundled under `${CLAUDE_SKILL_DIR}/references/` as local mirrors. Use these paths directly.

### `references/verified_large_kernel/` — Large kernel references
| File | Purpose |
|------|---------|
| `conv2d_3x64_b1a_fp32.cc` | Complete tiled conv2d kernel |
| `conv2d_3x64_b1a_fp32_test.py` | Full tiling + 4-core mapping + verification |
| `conv2d_3x64_b1a_fp32.py` | PyTorch reference model |

### `references/allo_examples/allo_tests/` — Allo dataflow/mapping patterns
| File | Purpose |
|------|---------|
| `test_norm.py` | Normalization test + ExternalModule pattern |
| `test_mapping_basic.py` | Basic mapping examples |
| `test_mapping_gemm.py` | GEMM mapping pattern |
| `test_meta_for.py` | Meta-programming (meta_if/meta_for) |
| `test_collective_communication.py` | Multi-core communication |
| `gemm.py` | GEMM end-to-end validation script |

### `references/allo_examples/` — Mirrored Allo reference bundle
| File | Purpose |
|------|---------|
| `allo_tests/*` | Test and mapping patterns |
| `allo_kernels/*` | AIE kernel implementations |

### `references/allo_examples/allo_kernels/` — AIE kernel implementations
| File | Purpose |
|------|---------|
| `norm.cc` | Vectorized normalization kernel |
| `mm.cc` | GEMM kernel implementation |
| `mixed_mm.cc` | Mixed-precision GEMM kernel implementation |
| `gelu.cc` | LUT-based activation kernel |
| `layer_norm.cc` | Normalization with vector ops |
| `softmax_bf16.cc` | Softmax bfloat16 |

### `references/api_doc/`
| File | Purpose |
|------|---------|
| `api_doc.md` | AMD AIE API — vector types, accumulators, memory ops, arithmetic |

### `references/allo_docs/`
| File | Purpose |
|------|---------|
| `dataflow.rst` | Allo dataflow design notes and examples |
| `memory.py` | Allo memory layout API source |

### In-skill documentation
- [examples.md](examples.md) — reference-first vectorized cookbook
