import os
import subprocess
from pathlib import Path

TOP_PRJ_ABS_DIR = "/home/qz425/llm_codegen/top.prj"

def analyze_trace(top_prj_dir, utils_dir=None, targetname="abs_int8", colshift=1, print_summary=True):
    top_prj_dir = Path(top_prj_dir)
    utils_dir = utils_dir or os.environ.get(
        "MLIR_AIE_UTILS_DIR", "/home/qz425/mlir-aie/programming_examples/utils"
    )

    trace_txt = top_prj_dir / "trace.txt"
    mlir_file = top_prj_dir / "top.mlir"
    if not trace_txt.exists() or not mlir_file.exists():
        print(f"Trace files not found in {top_prj_dir}. Skipping trace analysis.")
        return None

    parse_script = Path(utils_dir) / "parse_trace.py"
    summary_script = Path(utils_dir) / "get_trace_summary.py"
    json_out = top_prj_dir / f"trace_{targetname}.json"

    try:
        parse_proc = subprocess.run(
            [
                "python3",
                str(parse_script),
                "--filename",
                str(trace_txt),
                "--mlir",
                str(mlir_file),
                "--colshift",
                str(colshift),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(json_out, "w") as f:
            f.write(parse_proc.stdout)
        print(f"Wrote trace JSON: {json_out}")

        if print_summary:
            summary_proc = subprocess.run(
                [
                    "python3",
                    str(summary_script),
                    "--filename",
                    str(json_out),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(summary_proc.stdout)
    except Exception as e:
        print(f"Trace analysis failed: {e}")
        return None

    return json_out


