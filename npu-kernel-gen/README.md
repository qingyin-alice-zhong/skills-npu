# NPU Kernel Generation Skill

A Claude Code skill for generating AMD XDNA NPU kernel `.cc` files and allo `test.py` files.

## What it does

Given an operation name, data type, and tensor shapes, this skill generates complete, ready-to-test NPU kernel code following the patterns established in the [allo](https://github.com/cornell-zhang/allo) framework.

**Two modes:**

- **Small kernel** (single tile, ≤1024 elements/buffer): generates vectorized `kernel_func.cc` and `test.py`
- **Large kernel** (multi-tile, e.g. from PyTorch `nn.Module`): generates a tiled `.cc` kernel + `_test.py` with multi-core mapping, tile decomposition, and full-shape verification

## Usage

```
/npu-kernel-gen relu int8
/npu-kernel-gen sigmoid bfloat16 256
/npu-kernel-gen Conv2d(3, 64, kernel_size=3, padding=1) float32 input=[10,3,224,224]
```

Or describe the operation in natural language — the skill triggers automatically when you ask to create an NPU/AIE kernel.

## Structure

```
├── SKILL.md                    # Main skill instructions
├── examples.md                 # Reference-first cookbook (allo + mlir-aie)
└── references/
    ├── api_doc/                # AMD AIE API documentation
    ├── allo_docs/              # Mirrored Allo docs/utilities (dataflow.rst, memory.py)
    ├── allo_examples/          # Mirrored Allo reference bundle
    │   ├── allo_kernels/       # Kernel implementations (norm, mm, mixed_mm, ...)
    │   └── allo_tests/         # Dataflow mapping/tiling/test patterns (incl. gemm.py)
    └── verified_large_kernel/  # Large kernel tiling + 4-core mapping example
```

## Key features

- Automatic size classification (small vs. large kernel)
- Hardware-aware tile sizing (64KB local memory, 1024-element DMA limit)
- Type mapping for int8/int16/int32/bfloat16/float32 across C++, allo, and numpy
- Multi-core parallel dispatch with `pid`-based routing
- Reference-grounded generation — reads real working code before generating
- Reference paths are local-only — all examples are copied under `references/`
- Vectorization-first policy — prefers `aie::load_v/store_v` + pipeline pragmas
