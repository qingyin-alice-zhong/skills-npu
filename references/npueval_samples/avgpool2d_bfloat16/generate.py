# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor

np.random.seed(0)
description = "A kernel that performs a two dimension average pooling on an input matrix (32x32), buffer_in, with a 2x2 kernel and stride of two. The output is written to buffer_out."

def behavioral(input):
    rows, cols = input.shape
    output = np.zeros((rows//2, cols//2), dtype=input.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input[i:i+2, j:j+2]
            output[i//2, j//2] = np.mean(window)
    return output

input = np.random.randn(32, 32).astype(bfloat16) * 2

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[input],
    rtp_values=None
)

pc.write_json("kernel.json")