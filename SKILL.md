---
name: npu-kernel-gen
description: Generate AMD XDNA NPU kernel .cc files and allo test.py files for AIE operations. Use when the user asks to create a new NPU kernel, AIE kernel, wants to add a new operation to the npueval_dataset, or wants to convert a PyTorch operation to an NPU kernel. Handles both small single-tile kernels and large multi-tile kernels with tiling/mapping.
argument-hint: "<operation_name> <dtype> [buffer_sizes...] or <pytorch_op_description>"
---

# NPU Kernel & Test Generator

Generate NPU kernel `.cc` files and `test.py` for AIE operations. Supports two modes:
- **Small kernel** (data fits in single tile, <=1024 elements per buffer): generates `kernel_func.cc`, `canonical_scalar.cc`, `canonical_scalar_allo.cc`, `test.py`
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
├── kernel_func.cc              # AIE kernel stub (TODO placeholder)
├── canonical_scalar.cc         # Pure C++ scalar reference implementation
├── canonical_scalar_allo.cc    # AIE-wrapped scalar implementation
└── test.py                     # Allo dataflow test harness
```

### File Templates

#### 1. kernel_func.cc — AIE Kernel Stub

```cpp
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void KERNEL_NAME(PARAMS_WITH_FIXED_ARRAYS) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"
```

#### 2. canonical_scalar.cc — Pure C++ Reference

```cpp
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void KERNEL_NAME(PARAMS_WITH_POINTERS) {
    // Pure scalar C++ implementation
    // NO AIE includes, NO event0/event1
    // Use raw pointers (e.g., int8_t*, bfloat16*)
}
```

#### 3. canonical_scalar_allo.cc — AIE-Wrapped Scalar

```cpp
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void KERNEL_NAME(PARAMS_WITH_FIXED_ARRAYS) {
    event0();
    // Same scalar logic as canonical_scalar.cc
    // but with std:: prefixed types (std::int8_t, std::int32_t, std::uint32_t)
    event1();
}

} // extern "C"
```

#### 4. test.py — Allo Test Harness (Small)

```python
import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import ALLO_TYPE
# If bfloat16: from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = INPUT_BUFFER_SIZE

# Reference code starts
def reference_KERNEL_NAME(ANNOTATED_PARAMS) -> Annotated[np.ndarray, "shape: (OUTPUT_SIZE,)"]:
    # NumPy reference implementation
    pass
# Reference code ends


def _test_KERNEL_NAME(kernel_path: str):
    KERNEL_NAME_kernel = ExternalModule(
        top="KERNEL_NAME",
        impl_path=kernel_path,
        input_idx=INPUT_IDX_LIST,
        output_idx=OUTPUT_IDX_LIST,
    )

    Ty = ALLO_TYPE
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(KERNEL_PARAMS_WITH_LAYOUT):
            KERNEL_NAME_kernel(CALL_ARGS)

    # Generate random test inputs
    INPUT_GENERATION

    ref_output = reference_KERNEL_NAME(INPUT_ARGS)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top, target="aie", profile=True, warmup=5, num_iters=20,
            trace=[("core", (0,))], trace_size=655360, project=TOP_PRJ_ABS_DIR
        )
        output_allo = np.zeros((OUTPUT_SIZE,), dtype=NP_DTYPE)
        mod(MOD_CALL_ARGS)
        try:
            np.testing.assert_allclose(OUTPUT_COMPARE_LHS, OUTPUT_COMPARE_RHS, rtol=RTOL, atol=ATOL)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="KERNEL_NAME", colshift=1)
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)
    _test_KERNEL_NAME(args.kernel_path)
```

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

### .cc Template (Large Kernel)

The .cc operates on **one tile's worth of data**, with all tensors flattened to 1D:

```cpp
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void KERNEL_NAME(float input[INPUT_SIZE], float output[OUTPUT_SIZE], float param[PARAM_SIZE]) {
    // Tile-level constants (must match test.py)
    constexpr int TILE_DIM_A = ...;
    constexpr int TILE_DIM_B = ...;

    event0();

    // Unpack param buffer if needed (e.g., weights + bias)
    const float *weights = param;
    const float *bias = param + WEIGHT_SIZE;

    // Core computation on ONE tile — pure scalar loops
    for (...) {
        for (...) {
            // accumulate with double for float32 precision
            double acc = ...;
            for (...) {
                acc += ...;
            }
            output[idx] = static_cast<float>(acc);
        }
    }

    event1();
}

} // extern "C"
```

### test.py Template (Large Kernel with Tiling + Mapping)

```python
import os
import argparse
import shutil
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import allo
import allo.dataflow as df
from allo.ir.types import float32
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import TOP_PRJ_ABS_DIR

# ===================== Layout & Mapping =====================
S = Layout.Shard
R = Layout.Replicate
LyRep = [R]
MAPPING_CORES = 4   # Number of parallel AIE cores

# ===================== Full Problem Dimensions =====================
# (from PyTorch reference)
FULL_BATCH = ...
FULL_CHANNELS = ...
FULL_HEIGHT = ...
FULL_WIDTH = ...

# ===================== Tile Dimensions =====================
# CRITICAL: tile sizes chosen so each buffer fits in ~4KB
TILE_... = ...

# ===================== Per-Tile Buffer Sizes (flattened 1D) =====================
INPUT_SIZE  = ...   # Must match .cc parameter
OUTPUT_SIZE = ...   # Must match .cc parameter
PARAM_SIZE  = ...   # Must match .cc parameter (if any)

# Tolerance
NUMERIC_RTOL = 1e-2
NUMERIC_ATOL = 1e-2


def iter_tile_starts(full_extent: int, tile_extent: int):
    """Yield (start, actual_extent) for each tile along a dimension."""
    for start in range(0, full_extent, tile_extent):
        yield start, min(tile_extent, full_extent - start)


def build_test_case(seed=0):
    """Generate full-size random inputs + PyTorch reference output."""
    rng = np.random.default_rng(seed)
    # Generate inputs matching PyTorch shapes
    full_input = (rng.standard_normal(FULL_SHAPE, dtype=np.float32) * 0.05).astype(np.float32)
    full_weight = (rng.standard_normal(WEIGHT_SHAPE, dtype=np.float32) * 0.05).astype(np.float32)
    full_bias = (rng.standard_normal(BIAS_SHAPE, dtype=np.float32) * 0.02).astype(np.float32)

    # Compute PyTorch reference
    with torch.no_grad():
        ref_output = PYTORCH_OP(
            torch.from_numpy(full_input),
            torch.from_numpy(full_weight),
            torch.from_numpy(full_bias),
        ).cpu().numpy()

    return full_input, full_weight, full_bias, ref_output


def extract_tile(full_input, full_weight, full_bias, tile_coords):
    """Extract one tile's input/param from full tensors.

    MUST handle:
    - Padding/halo for convolutions
    - Boundary tiles (may be smaller than TILE_SIZE)
    - Weight slicing for tiled output channels
    - Flattening to 1D arrays matching .cc signature
    """
    ...
    input_flat = input_patch.reshape(-1).astype(np.float32)
    param_flat = np.concatenate([weight_patch.reshape(-1), bias_patch.reshape(-1)])
    return input_flat, param_flat


def run_tiled(mod, full_input, full_weight, full_bias, ref_output):
    """Tile decomposition loop: extract tiles → dispatch to NPU → reassemble."""
    # Pad input if needed (e.g., for conv)
    padded_input = np.pad(full_input, PAD_SPEC, mode="constant", constant_values=0.0)

    output = np.zeros(FULL_OUTPUT_SHAPE, dtype=np.float32)
    mismatch_count = 0

    # Build tile task list
    tile_tasks = []
    for n in range(FULL_BATCH):
        for dim1_start, dim1_extent in iter_tile_starts(FULL_DIM1, TILE_DIM1):
            for dim2_start, dim2_extent in iter_tile_starts(FULL_DIM2, TILE_DIM2):
                tile_tasks.append((n, dim1_start, dim1_extent, dim2_start, dim2_extent))

    # Process in groups of MAPPING_CORES
    for group_start in range(0, len(tile_tasks), MAPPING_CORES):
        group = tile_tasks[group_start:group_start + MAPPING_CORES]

        # Prepare per-lane buffers
        input_group = np.zeros((MAPPING_CORES, INPUT_SIZE), dtype=np.float32)
        param_group = np.zeros((MAPPING_CORES, PARAM_SIZE), dtype=np.float32)

        for lane, task in enumerate(group):
            input_group[lane], param_group[lane] = extract_tile(
                padded_input, full_weight, full_bias, task
            )

        # Run on NPU (all MAPPING_CORES lanes in parallel)
        output_group = np.zeros((MAPPING_CORES, OUTPUT_SIZE), dtype=np.float32)
        mod(
            input_group[0], input_group[1], input_group[2], input_group[3],
            output_group[0], output_group[1], output_group[2], output_group[3],
            param_group[0], param_group[1], param_group[2], param_group[3],
        )

        # Write back tiles to full output + verify each tile
        for lane, task in enumerate(group):
            tile_output = output_group[lane].reshape(TILE_OUTPUT_SHAPE)
            expected = ref_output[TILE_SLICE]  # slice matching this tile
            if not np.allclose(tile_output, expected, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL):
                mismatch_count += 1
            output[TILE_SLICE] = tile_output

    return output, mismatch_count


def _test_KERNEL_NAME(kernel_path: str):
    kernel = ExternalModule(
        top="KERNEL_NAME",
        impl_path=kernel_path,
        input_idx=[0, 2],    # input, param
        output_idx=[1],       # output
    )

    # Multi-core dataflow region: each core gets its own buffers
    @df.region()
    def top(
        A0: float32[INPUT_SIZE], A1: float32[INPUT_SIZE],
        A2: float32[INPUT_SIZE], A3: float32[INPUT_SIZE],
        C0: float32[OUTPUT_SIZE], C1: float32[OUTPUT_SIZE],
        C2: float32[OUTPUT_SIZE], C3: float32[OUTPUT_SIZE],
        P0: float32[PARAM_SIZE], P1: float32[PARAM_SIZE],
        P2: float32[PARAM_SIZE], P3: float32[PARAM_SIZE],
    ):
        @df.kernel(
            mapping=[MAPPING_CORES],
            args=[A0, A1, A2, A3, C0, C1, C2, C3, P0, P1, P2, P3],
        )
        def core(
            lA0: float32[INPUT_SIZE] @ LyRep, lA1: float32[INPUT_SIZE] @ LyRep,
            lA2: float32[INPUT_SIZE] @ LyRep, lA3: float32[INPUT_SIZE] @ LyRep,
            lC0: float32[OUTPUT_SIZE] @ LyRep, lC1: float32[OUTPUT_SIZE] @ LyRep,
            lC2: float32[OUTPUT_SIZE] @ LyRep, lC3: float32[OUTPUT_SIZE] @ LyRep,
            lP0: float32[PARAM_SIZE] @ LyRep, lP1: float32[PARAM_SIZE] @ LyRep,
            lP2: float32[PARAM_SIZE] @ LyRep, lP3: float32[PARAM_SIZE] @ LyRep,
        ):
            pid, = df.get_pid()
            with allo.meta_if(pid == 0):
                kernel(lA0, lC0, lP0)
            with allo.meta_elif(pid == 1):
                kernel(lA1, lC1, lP1)
            with allo.meta_elif(pid == 2):
                kernel(lA2, lC2, lP2)
            with allo.meta_else():
                kernel(lA3, lC3, lP3)

    full_input, full_weight, full_bias, ref_output = build_test_case(seed=0)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(top, target="aie", profile=False, warmup=0, num_iters=1, project=TOP_PRJ_ABS_DIR)
        output_allo, mismatch_count = run_tiled(mod, full_input, full_weight, full_bias, ref_output)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=NUMERIC_RTOL, atol=NUMERIC_ATOL)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")
            if mismatch_count > 0:
                print(f"Mismatched tiles: {mismatch_count}")
    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)
    _test_KERNEL_NAME(args.kernel_path)
```

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

| dtype      | kernel_func.cc / canonical_scalar_allo.cc | canonical_scalar.cc |
|------------|-------------------------------------------|---------------------|
| int8       | `std::int8_t`                             | `int8_t`            |
| int16      | `std::int16_t`                            | `int16_t`           |
| int32      | `std::int32_t`                            | `int32_t`           |
| bfloat16   | `bfloat16`                                | `bfloat16`          |
| float32    | `float`                                   | `float`             |

- `kernel_func.cc` and `canonical_scalar_allo.cc`: parameters use **fixed-size arrays** (e.g., `std::int8_t in[1024]`)
- `canonical_scalar.cc`: parameters use **raw pointers** (e.g., `int8_t* in`)
- Loop variables in AIE-wrapped files use `std::uint32_t` / `std::int32_t`

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

## Workflow (MUST follow this order)

### Step 1: Parse & Classify

1. **Parse** the operation semantics from user input (PyTorch op, shapes, dtype)
2. **Classify** as small or large kernel based on buffer sizes (see Step 0 above)

### Step 2: Read Reference Files (MANDATORY before generating any code)

**You MUST read real reference files from the codebase before writing any .cc or test.py.** Do NOT rely solely on templates in this skill — always ground your output in actual working code from the repository.

Choose the most similar existing kernel based on operation type:

#### For Small Kernels — read from `llm_codegen_spec/npueval_dataset/`

Select the closest match by operation category, then READ all files from the **bundled references** inside this skill:

| Your operation type | Reference to read (under `${CLAUDE_SKILL_DIR}/references/npueval_samples/`) | Why |
|--------------------|-----------------------|-----|
| Element-wise unary (relu, abs, sign, ceil, sqrt, negate) | `relu_int8/` | Simple 1-in-1-out pattern |
| Element-wise unary float (sigmoid, tanh, gelu) | `sigmoid_bfloat16/` | Math-heavy, bfloat16 handling |
| Element-wise binary (add, subtract, min, max) | `vectoradd_bfloat16/` | 2-in-1-out pattern |
| Pooling (avgpool, maxpool) | `avgpool2d_bfloat16/` | 2D spatial, output != input size |
| MatMul / dot product | `matmul_16x16_int8/` | Multi-loop accumulation |
| Conv1d | `conv1d_bfloat16/` | Sliding window with params |

**Concrete read steps** (example for element-wise unary bfloat16):
```
Read: ${CLAUDE_SKILL_DIR}/references/npueval_samples/sigmoid_bfloat16/kernel_func.cc
Read: ${CLAUDE_SKILL_DIR}/references/npueval_samples/sigmoid_bfloat16/canonical_scalar.cc
Read: ${CLAUDE_SKILL_DIR}/references/npueval_samples/sigmoid_bfloat16/canonical_scalar_allo.cc
Read: ${CLAUDE_SKILL_DIR}/references/npueval_samples/sigmoid_bfloat16/test.py
```

If the bundled samples don't cover your exact case, also search `llm_codegen_spec/npueval_dataset/` for a closer match.

#### For Large Kernels — read from bundled `${CLAUDE_SKILL_DIR}/references/large_kernel/`

**Always read these files:**
```
Read: ${CLAUDE_SKILL_DIR}/references/large_kernel/conv2d_3x64_b1a_fp32.cc
Read: ${CLAUDE_SKILL_DIR}/references/large_kernel/conv2d_3x64_b1a_fp32_test.py
Read: ${CLAUDE_SKILL_DIR}/references/large_kernel/conv2d_3x64_b1a_fp32.py
```

**Also read mapping/layout references from bundled `${CLAUDE_SKILL_DIR}/references/allo_tests/`** (pick the most relevant):

| Your operation type | Additional reference to read |
|--------------------|------------------------------|
| Conv / spatial ops | `allo_tests/test_norm.py` + `allo_tests/norm.cc` |
| MatMul / GEMM | `allo_tests/test_mapping_gemm.py` |
| General mapping patterns | `allo_tests/test_mapping_basic.py` |
| Multi-core parallel | `allo_tests/test_collective_communication.py` |
| Meta-programming (meta_if/meta_for) | `allo_tests/test_meta_for.py` |

#### For ANY kernel — optionally read from bundled references:
- `${CLAUDE_SKILL_DIR}/references/api_doc/api_doc.md` — AIE vector/accumulator API details
- `${CLAUDE_SKILL_DIR}/references/allo_kernels/memory.py` — Layout/Shard/Replicate API
- `${CLAUDE_SKILL_DIR}/references/allo_kernels/gelu.cc` — LUT-based activation kernel example
- `${CLAUDE_SKILL_DIR}/references/allo_kernels/layer_norm.cc` — normalization kernel example
- `${CLAUDE_SKILL_DIR}/references/allo_kernels/softmax_bf16.cc` — softmax bfloat16 example

### Step 3: Plan Tile Geometry (large kernels only)

1. **Calculate** tile dimensions respecting hardware constraints
2. **Verify memory budget**: `(INPUT_SIZE + OUTPUT_SIZE + PARAM_SIZE) * dtype_bytes < 64KB`
3. **Report** tile geometry to user before generating code

### Step 4: Generate Code

Generate files by **adapting the reference code you just read**, NOT by filling in abstract templates:

1. **Start from the reference test.py** you read in Step 2
2. **Modify** the reference to match the new operation's semantics:
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
  - Small kernel: `python test.py --kernel_path canonical_scalar_allo.cc`
  - Large kernel: `python main.py --kernel {name}`

## Bundled Reference Files Index

All references are bundled under `${CLAUDE_SKILL_DIR}/references/`. Read from here instead of searching the repo.

### `references/npueval_samples/` — Small kernel references
| Directory | Pattern |
|-----------|---------|
| `relu_int8/` | Simplest unary int8 (4 files: kernel_func.cc, canonical_scalar.cc, canonical_scalar_allo.cc, test.py) |
| `sigmoid_bfloat16/` | Unary bfloat16 with math |
| `vectoradd_bfloat16/` | Binary operation bfloat16 |
| `matmul_16x16_int8/` | MatMul with accumulator |
| `avgpool2d_bfloat16/` | Different input/output sizes |
| `conv1d_bfloat16/` | Conv with params buffer |
| `utils.py` | Shared utilities (TOP_PRJ_ABS_DIR, analyze_trace) |

### `references/large_kernel/` — Large kernel references
| File | Purpose |
|------|---------|
| `conv2d_3x64_b1a_fp32.cc` | Complete tiled conv2d kernel |
| `conv2d_3x64_b1a_fp32_test.py` | Full tiling + 4-core mapping + verification |
| `conv2d_3x64_b1a_fp32.py` | PyTorch reference model |
| `abs_int8.cc` | Simple vectorized AIE kernel |
| `abs_int8_test.py` | Simple kernel test with mapping=[1] |

### `references/allo_tests/` — Allo dataflow/mapping patterns
| File | Purpose |
|------|---------|
| `test_norm.py` + `norm.cc` | Normalization tiling pattern |
| `test_mapping_basic.py` | Basic mapping examples |
| `test_mapping_gemm.py` | GEMM mapping pattern |
| `test_meta_for.py` | Meta-programming (meta_if/meta_for) |
| `test_collective_communication.py` | Multi-core communication |

### `references/allo_kernels/` — AIE kernel implementations
| File | Purpose |
|------|---------|
| `gelu.cc` | LUT-based activation kernel |
| `layer_norm.cc` | Normalization with vector ops |
| `softmax_bf16.cc` | Softmax bfloat16 |
| `memory.py` | Layout/Shard/Replicate API source |

### `references/api_doc/`
| File | Purpose |
|------|---------|
| `api_doc.md` | AMD AIE API — vector types, accumulators, memory ops, arithmetic |

### In-skill documentation
- [examples.md](examples.md) — 5 annotated examples with explanations (small + large kernels)
