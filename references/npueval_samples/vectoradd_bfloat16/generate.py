# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel computes the elementwise addition of two bfloat16 input vectors of size 256, writing the result to an output bfloat16 vector."

def behavioral(in0, in1):
    return (in0 + in1).astype(bfloat16)

in0 = (np.random.randn(256)).astype(bfloat16)
in1 = (np.random.randn(256)).astype(bfloat16)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in0, in1],
    rtp_values=None
)

pc.write_json("kernel.json")