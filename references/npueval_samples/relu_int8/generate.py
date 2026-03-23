# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from npueval.datasetbuilder import PromptConstructor
np.random.seed(0)

description = "This AIE kernel performs a ReLU activation on an int8 input vector of size num_elements."

def behavioral(in_buffer):
    return np.where(in_buffer > 0, in_buffer, 0).astype(np.int8)

in_buffer = np.random.randint(-128, 127, size=(1024,), dtype=np.int8)

pc = PromptConstructor(
    source_path="canonical_scalar.cc",
    description=description,
    behavioral=behavioral,
    input_arrays=[in_buffer],
    rtp_values=None
)

pc.write_json("kernel.json")