import os
import argparse
from typing import Annotated

import numpy as np
import shutil
from pathlib import Path

import allo.dataflow as df
from allo.ir.types import bfloat16
from ml_dtypes import bfloat16 as np_bfloat16
from allo.memory import Layout
from allo.backend.aie.external_kernel import ExternalModule

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 1024  # 32x32 input flattened

# Reference code starts
def reference_avgpool2d_bfloat16(input_flat: Annotated[np.ndarray, "shape: (1024,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    # Reshape flat input to 32x32 matrix
    input_2d = input_flat.reshape((32, 32))
    rows, cols = input_2d.shape
    output = np.zeros((rows//2, cols//2), dtype=input_2d.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input_2d[i:i+2, j:j+2]
            output[i//2, j//2] = np.mean(window)
    return output.flatten().astype(np_bfloat16)
# Reference code ends


def _test_avgpool2d_bfloat16(kernel_path: str):
    avgpool2d_bfloat16_kernel = ExternalModule(
        top="avgpool2d_bfloat16",
        impl_path=kernel_path,
        input_idx=[0],
        output_idx=[1],
    )

    Ty = bfloat16
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: bfloat16[M] @ Ly, B: bfloat16[256] @ Ly):
            avgpool2d_bfloat16_kernel(A, B)

    # Generate random bfloat16 input for 32x32 matrix
    input_tensor = (np.random.randn(1024) * 2).astype(np_bfloat16)
    
    ref_output = reference_avgpool2d_bfloat16(input_tensor)

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
        output_allo = np.zeros((256,), dtype=np_bfloat16)
        mod(input_tensor, output_allo)
        try:
            # Convert bfloat16 to float32 for comparison
            np.testing.assert_allclose(output_allo.astype(np.float32), ref_output.astype(np.float32), rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="avgpool2d_bfloat16", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_avgpool2d_bfloat16(args.kernel_path)
