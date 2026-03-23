# NPU Kernel Generation Examples

These are real examples from the codebase for reference.

## Example 1: Simple Element-wise — relu_int8

### kernel_func.cc
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

void relu_int8(std::int8_t in_buffer[1024], std::int8_t out_buffer[1024]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"
```

### canonical_scalar.cc
```cpp
// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void relu_int8(int8_t *in_buffer, int8_t* out_buffer) {
    constexpr int32_t num_elements = 1024;
    for (uint32_t i = 0; i < num_elements; i++) {
        int8_t v = in_buffer[i];
        out_buffer[i] = (v > 0) ? v : 0;
    }
}
```

### canonical_scalar_allo.cc
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

void relu_int8(std::int8_t in_buffer[1024], std::int8_t out_buffer[1024]) {
    event0();
    constexpr std::int32_t num_elements = 1024;
    for (std::uint32_t i = 0; i < num_elements; i++) {
        std::int8_t v = in_buffer[i];
        out_buffer[i] = (v > 0) ? v : 0;
    }
    event1();
}

} // extern "C"
```

### test.py
```python
import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import int8
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 1024

# Reference code starts
def reference_relu_int8(x: Annotated[np.ndarray, "shape: (1024,)"]) -> Annotated[np.ndarray, "shape: (1024,)"]:
    return np.where(x > 0, x, 0).astype(np.int8)
# Reference code ends


def _test_relu_int8(kernel_path: str):
    relu_int8_kernel = ExternalModule(
        top="relu_int8",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = int8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int8[M] @ Ly, B: int8[M] @ Ly):
            relu_int8_kernel(A, B)

    input_tensor = np.random.randint(-100, 100, (1024,), dtype=np.int8)
    ref_output = reference_relu_int8(input_tensor)

    if "MLIR_AIE_INSTALL_DIR" in os.environ:
        mod = df.build(
            top,
            target="aie",
            profile=True,
            warmup=5,
            num_iters=20,
            trace=[("core", (0,))],
            trace_size=655360,
            project=TOP_PRJ_ABS_DIR
        )
        output_allo = np.zeros((tensor_size,), dtype=np.int8)
        mod(input_tensor, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="relu_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_relu_int8(args.kernel_path)
```

---

## Example 2: Multi-input — vectoradd_bfloat16

Key differences from single-input:
- **2 inputs + 1 output** → `input_idx=[0, 1], output_idx=[2]`
- bfloat16 requires `from ml_dtypes import bfloat16 as np_bfloat16`
- Comparison converts to float32: `output.astype(np.float32)`

### kernel_func.cc
```cpp
void vectoradd_bfloat16(bfloat16 in0[256], bfloat16 in1[256], bfloat16 out[256]) {
    event0();
    // TODO: Implement the kernel
    event1();
}
```

### canonical_scalar.cc
```cpp
void vectoradd_bfloat16(bfloat16 *in0, bfloat16 *in1, bfloat16 *out) {
    constexpr int32_t N = 256;
    for (int i = 0; i < N; i++) {
        out[i] = in0[i] + in1[i];
    }
}
```

### test.py (key parts)
```python
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16

# Reference code starts
def reference_vectoradd_bfloat16(vec_a: Annotated[np.ndarray, "shape: (256,)"], vec_b: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    return (vec_a + vec_b).astype(np_bfloat16)
# Reference code ends

# ExternalModule: input_idx=[0, 1], output_idx=[2]
# core(A: bfloat16[M] @ Ly, B: bfloat16[M] @ Ly, C: bfloat16[M] @ Ly)
# assert_allclose uses .astype(np.float32) on both sides, rtol=1e-2, atol=1e-2
```

---

## Example 3: Different input/output sizes — avgpool2d_bfloat16

Key differences:
- Input 1024 (32x32 flattened), Output 256 (16x16 flattened)
- Output buffer size differs from input

### kernel_func.cc
```cpp
void avgpool2d_bfloat16(bfloat16 input[1024], bfloat16 output[256]) {
    event0();
    // TODO: Implement the kernel
    event1();
}
```

### test.py (key parts)
```python
# core(A: bfloat16[M] @ Ly, B: bfloat16[256] @ Ly)  — note different sizes
# output_allo = np.zeros((256,), dtype=np_bfloat16)
```

---

## Example 4: Matrix multiplication — matmul_16x16_int8

Key differences:
- 2 inputs + 1 output, all 256 elements (16x16 flattened)
- Uses int32 accumulator with clamping to int8 range
- Small random range `(-10, 10)` to avoid overflow

### canonical_scalar.cc
```cpp
void matmul_16x16_int8(int8_t* a, int8_t* b, int8_t* out) {
    constexpr int SIZE = 16;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < SIZE; ++k) {
                acc += static_cast<int32_t>(a[i*SIZE + k]) * static_cast<int32_t>(b[k*SIZE + j]);
            }
            if (acc > 127) acc = 127;
            if (acc < -128) acc = -128;
            out[i*SIZE + j] = static_cast<int8_t>(acc);
        }
    }
}
```

### test.py (key parts)
```python
# reference uses np.matmul with int32 intermediate + np.clip(-128, 127)
# Random: np.random.randint(-10, 10, (256,), dtype=np.int8)
```

---

## Example 5: Large Kernel with Tiling — Conv2d [10,3,224,224] → [10,64,224,224]

This is a real VGG16 first-layer conv2d. Full tensors are WAY too large for one tile, so we decompose.

**PyTorch reference**: `nn.Conv2d(3, 64, kernel_size=3, padding=1)`
- Input: `[10, 3, 224, 224]` = 1,505,280 float32 elements (~5.7MB)
- Output: `[10, 64, 224, 224]` = 32,112,640 float32 elements (~122MB)
- Weights: `[64, 3, 3, 3]` + bias `[64]` = 1,792 float32 elements

### Tile Design

```python
TILE_OUT_CHANNELS = 8       # Process 8 of 64 output channels per tile
TILE_OUT_HEIGHT = 8          # 8 rows of output per tile
TILE_OUT_WIDTH = 8           # 8 columns per tile

# Input tile includes convolution halo
TILE_INPUT_HEIGHT = 8 + 3 - 1 = 10    # output_h + kernel_h - 1
TILE_INPUT_WIDTH  = 8 + 3 - 1 = 10

# Per-tile buffer sizes (flattened 1D)
INPUT_SIZE  = 3 * 10 * 10 = 300   →  1.2 KB
OUTPUT_SIZE = 8 * 8 * 8   = 512   →  2.0 KB
PARAM_SIZE  = (8*3*3*3) + 8 = 224 →  0.9 KB
# Total per tile: ~4.1 KB ✓ (fits in 64KB local memory)

# Total tiles: 10 × (64/8) × (224/8) × (224/8) = 62,720
# With MAPPING_CORES=4: 15,680 groups
```

### .cc (tile-level kernel)
```cpp
void conv2d_3x64_b1a_fp32(float input[300], float output[512], float param[224]) {
    constexpr int IN_CHANNELS = 3;
    constexpr int OUT_CHANNELS = 8;       // tile, not full 64
    constexpr int INPUT_HEIGHT = 10;       // includes halo
    constexpr int INPUT_WIDTH = 10;
    constexpr int OUTPUT_HEIGHT = 8;
    constexpr int OUTPUT_WIDTH = 8;
    constexpr int KERNEL_H = 3;
    constexpr int KERNEL_W = 3;
    constexpr int WEIGHT_SIZE = OUT_CHANNELS * IN_CHANNELS * KERNEL_H * KERNEL_W;

    event0();

    const float *in = input;
    const float *weights = param;
    const float *bias = param + WEIGHT_SIZE;   // param = [weights..., bias...]

    for (int oc = 0; oc < OUT_CHANNELS; ++oc) {
        for (int oh = 0; oh < OUTPUT_HEIGHT; ++oh) {
            for (int ow = 0; ow < OUTPUT_WIDTH; ++ow) {
                double acc = static_cast<double>(bias[oc]);  // double for precision
                for (int ic = 0; ic < IN_CHANNELS; ++ic) {
                    for (int kh = 0; kh < KERNEL_H; ++kh) {
                        for (int kw = 0; kw < KERNEL_W; ++kw) {
                            const int ih = oh + kh;
                            const int iw = ow + kw;
                            acc += static_cast<double>(in[(ic*INPUT_HEIGHT + ih)*INPUT_WIDTH + iw])
                                 * static_cast<double>(weights[((oc*IN_CHANNELS + ic)*KERNEL_H + kh)*KERNEL_W + kw]);
                        }
                    }
                }
                output[(oc*OUTPUT_HEIGHT + oh)*OUTPUT_WIDTH + ow] = static_cast<float>(acc);
            }
        }
    }

    event1();
}
```

### test.py (key patterns for large kernel)

**Multi-core dataflow region (4 cores, explicit per-core buffers):**
```python
MAPPING_CORES = 4

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
```

**Tile extraction (with padding and weight slicing):**
```python
# 1. Pad full input BEFORE tiling
padded_input = np.pad(full_input, ((0,0),(0,0),(PADDING,PADDING),(PADDING,PADDING)), mode="constant")

# 2. For each tile, extract the input patch with halo
input_patch = padded_input[n, :, h_start:h_start+TILE_INPUT_HEIGHT, w_start:w_start+TILE_INPUT_WIDTH]

# 3. Slice weights for this tile's output channels
weight_patch = full_weight[oc_start:oc_start+oc_extent]  # may need zero-padding if partial

# 4. Pack into param buffer
param = np.concatenate([weight_patch.reshape(-1), bias_patch.reshape(-1)])
```

**Group dispatch (process MAPPING_CORES tiles at once):**
```python
for group_start in range(0, len(tile_tasks), MAPPING_CORES):
    group = tile_tasks[group_start:group_start + MAPPING_CORES]
    input_group = np.zeros((MAPPING_CORES, INPUT_SIZE), dtype=np.float32)
    param_group = np.zeros((MAPPING_CORES, PARAM_SIZE), dtype=np.float32)
    output_group = np.zeros((MAPPING_CORES, OUTPUT_SIZE), dtype=np.float32)

    for lane, task in enumerate(group):
        input_group[lane], param_group[lane] = extract_tile(...)

    # Call NPU — all 4 lanes in parallel
    mod(
        input_group[0], input_group[1], input_group[2], input_group[3],
        output_group[0], output_group[1], output_group[2], output_group[3],
        param_group[0], param_group[1], param_group[2], param_group[3],
    )

    # Verify each tile and write back to full output
    for lane, task in enumerate(group):
        # ... compare with ref_output slice, write to output array
```
