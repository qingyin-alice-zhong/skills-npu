import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16, int32
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 256

# Reference code starts
def reference_conv1d_bfloat16(in_buffer: Annotated[np.ndarray, "shape: (256,)"], param: Annotated[np.ndarray, "shape: (3,)"]) -> Annotated[np.ndarray, "shape: (254,)"]:
    # Extract kernel and stride from param (param contains float32 values as int32)
    kernel_val0 = np.array([param[0]], dtype=np.int32).view(np.float32)[0]
    kernel_val1 = np.array([param[1]], dtype=np.int32).view(np.float32)[0]
    kernel = np.array([kernel_val0, kernel_val1], dtype=np_bfloat16)
    stride = int(param[2])
    
    vector_size = in_buffer.shape[0]
    kernel_size = kernel.shape[0]
    output_size = min((vector_size - kernel_size) // stride + 1, 254)  # Limit to 254 as expected by kernel
    out_buffer = np.zeros(output_size, dtype=np_bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel_size):
            acc += float(in_buffer[i * stride + j]) * float(kernel[j])
        out_buffer[i] = np_bfloat16(acc)
    return out_buffer
# Reference code ends


def _test_conv1d_bfloat16(kernel_path: str):
    conv1d_bfloat16_kernel = ExternalModule(
        top="conv1d_bfloat16",
        impl_path=kernel_path,
        input_idx=[0, 2],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, C: bfloat16[254] @ Ly, Param: int32[3] @ Ly):
            conv1d_bfloat16_kernel(A, C, Param)

    # Generate random bfloat16 input and parameters
    in_buffer = (np.random.randn(256) * 3.2).astype(np_bfloat16)
    
    # Pack kernel (2 values) and stride (1 value) into param array
    kernel_vals = (np.random.randn(2) * 0.5).astype(np.float32)
    stride_val = 1  # Use stride=1, output size = (256-2)/1+1 = 255, but we allocate 254
    # Pack kernel as float32 values and stride into int32 array
    param = np.array([kernel_vals[0].view(np.int32), kernel_vals[1].view(np.int32), stride_val], dtype=np.int32)
    
    ref_output = reference_conv1d_bfloat16(in_buffer, param.astype(np.float32))

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
        output_allo = np.zeros((254,), dtype=np_bfloat16)
        mod(in_buffer, output_allo, param)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="conv1d_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_conv1d_bfloat16(args.kernel_path)
