# NPU Kernel Generation Skill

A Claude Code skill for generating AMD XDNA NPU kernel `.cc` files and allo `test.py` files.

## What it does

Given an operation name, data type, and tensor shapes, this skill generates complete, ready-to-test NPU kernel code following the patterns established in the [allo](https://github.com/cornell-zhang/allo) framework.

**Two modes:**

- **Small kernel** (single tile, ≤1024 elements/buffer): generates `kernel_func.cc`, `canonical_scalar.cc`, `canonical_scalar_allo.cc`, and `test.py`
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
├── examples.md                 # 5 annotated examples (small + large kernels)
└── references/
    ├── api_doc/                # AMD AIE API documentation
    ├── allo_kernels/           # Kernel implementations (gelu, layer_norm, softmax)
    ├── allo_tests/             # Dataflow mapping/tiling patterns
    ├── npueval_samples/        # 6 complete small kernel examples
    │   ├── relu_int8/
    │   ├── sigmoid_bfloat16/
    │   ├── vectoradd_bfloat16/
    │   ├── matmul_16x16_int8/
    │   ├── avgpool2d_bfloat16/
    │   └── conv1d_bfloat16/
    └── large_kernel/           # Large kernel tiling + 4-core mapping example
```

## Key features

- Automatic size classification (small vs. large kernel)
- Hardware-aware tile sizing (64KB local memory, 1024-element DMA limit)
- Type mapping for int8/int16/int32/bfloat16/float32 across C++, allo, and numpy
- Multi-core parallel dispatch with `pid`-based routing
- Reference-grounded generation — reads real working code before generating
