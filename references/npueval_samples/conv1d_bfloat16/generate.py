# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "A kernel that performs a 1D convolution operation on a bfloat16 input vector with a bfloat16 kernel and given stride (runtime parameter). All vectors are 256-wide and the kernel is size 2."

def behavioral(in_buffer, kernel, stride):
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    output_size = (vector_size - kernel_size) // stride + 1
    out_buffer = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel_size):
            acc += float(in_buffer[i * stride + j]) * float(kernel[j])
        out_buffer[i] = bfloat16(acc)
    return out_buffer

in_buffer = np.random.randn(256).astype(bfloat16)
kernel = np.random.randn(2).astype(bfloat16)
stride = 1

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer, kernel],
    rtp_values=[stride]
)

pc.write_json("kernel.json")