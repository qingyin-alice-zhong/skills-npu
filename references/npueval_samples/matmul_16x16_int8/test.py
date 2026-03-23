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

# Analyze trace via shared utility if generated under top.prj/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import analyze_trace
from utils import TOP_PRJ_ABS_DIR

Ly = Layout("R")
tensor_size = 256  # 16x16 = 256

# Reference code starts
def reference_matmul_16x16_int8(a: Annotated[np.ndarray, "shape: (256,)"], b: Annotated[np.ndarray, "shape: (256,)"]) -> Annotated[np.ndarray, "shape: (256,)"]:
    # Reshape flat arrays to 16x16 matrices
    a_mat = a.reshape(16, 16)
    b_mat = b.reshape(16, 16)
    # Matrix multiplication with int32 intermediate
    res = np.matmul(a_mat.astype(np.int32), b_mat.astype(np.int32))
    # Clamp to int8 range and flatten
    res = np.clip(res, -128, 127).astype(np.int8)
    return res.flatten()
# Reference code ends


def _test_matmul_16x16_int8(kernel_path: str):
    matmul_16x16_int8_kernel = ExternalModule(
        top="matmul_16x16_int8",
        impl_path=kernel_path,
        input_idx=[0, 1],
        output_idx=[2],
    )

    Ty = int8
    M = tensor_size

    @df.region()
    def top():
        @df.kernel(mapping=[1])
        def core(A: int8[M] @ Ly, B: int8[M] @ Ly, C: int8[M] @ Ly):
            matmul_16x16_int8_kernel(A, B, C)

    matrix_a = np.random.randint(-10, 10, (256,), dtype=np.int8)  # 16x16 flattened
    matrix_b = np.random.randint(-10, 10, (256,), dtype=np.int8)  # 16x16 flattened

    ref_output = reference_matmul_16x16_int8(matrix_a, matrix_b)

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
        output_allo = np.zeros((256,), dtype=np.int8)
        mod(matrix_a, matrix_b, output_allo)
        try:
            np.testing.assert_allclose(output_allo, ref_output, rtol=1e-2, atol=1e-2)
            print("PASS!")
        except AssertionError as e:
            print("FAIL!")
            print(f"Verification failed:\n{str(e)}")

        # ===== Analyze trace via shared utility if generated under top.prj/ =====
        analyze_trace(top_prj_dir=TOP_PRJ_ABS_DIR, targetname="matmul_16x16_int8", colshift=1)

    else:
        print("MLIR_AIE_INSTALL_DIR unset. Skipping AIE backend test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel_path", type=str, default="canonical_scalar_allo.cc")
    args = parser.parse_args()

    # clean the top.prj/ directory if it exists
    if Path(TOP_PRJ_ABS_DIR).exists():
        shutil.rmtree(TOP_PRJ_ABS_DIR)

    _test_matmul_16x16_int8(args.kernel_path)
