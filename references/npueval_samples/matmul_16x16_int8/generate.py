# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = (
    "This kernel performs matrix multiplication between two 16x16 int8_t matrices A and B "
    "(row-major, shape=(16,16)), writing the result (clamped to int8_t) into a 16x16 int8_t output buffer (row-major)."
)

def behavioral(a, b):
    # Matrix multiplication for int8, result clamped to int8.
    res = np.matmul(a.astype(np.int32), b.astype(np.int32))
    res = np.clip(res, -128, 127).astype(np.int8)
    return res

a = np.random.randint(-10, 10, size=(16, 16), dtype=np.int8)
b = np.random.randint(-10, 10, size=(16, 16), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[a, b],
    rtp_values=None
)

pc.write_json("kernel.json")