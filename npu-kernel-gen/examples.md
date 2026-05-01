# NPU Kernel Generation Examples (Reference-First)

This document is intentionally **reference-driven**.  
Do not treat these as invented templates. Start from mirrored code under `references/`, then adapt minimally.

---

## 0) Source of Truth Order

When generating new kernels/tests, read references in this order:

1. `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/*` (ground truth for `ExternalModule`, mapping, test style)
2. `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/*` (ground truth for vectorized kernel coding style)
3. `${CLAUDE_SKILL_DIR}/references/allo_examples/*` (end-to-end build/run patterns)
4. `https://github.com/Xilinx/mlir-aie/tree/main/programming_guide` (vectorization + optimization rules)
5. `${CLAUDE_SKILL_DIR}/references/verified_large_kernel/*` for task-specific patterns

---

## 1) Small Kernel Pattern (Unary/Binary/Pool/Matmul)

### Primary references

- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_norm.py`
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_kernels/norm.cc`

### Mandatory structure

- Python side keeps `ExternalModule(...)` + `@df.region()` + `@df.kernel(mapping=[1])`
- C++ side keeps `extern "C"`, `event0()`, `event1()`
- Numpy/PyTorch reference stays in Python and is compared with `np.testing.assert_allclose`

---

## 2) Large Kernel Pattern (Tiling + Multi-core)

### Primary references

- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/gemm.py`
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_mapping_gemm.py`
- `${CLAUDE_SKILL_DIR}/references/allo_examples/allo_tests/test_collective_communication.py`
- `${CLAUDE_SKILL_DIR}/references/verified_large_kernel/conv2d_3x64_b1a_fp32_test.py`

### Mandatory structure

- Test constructs tile tasks and dispatches in groups of `MAPPING_CORES`
- Uses `df.get_pid()` with `allo.meta_if/meta_elif/meta_else` for lane routing
- Keeps per-core buffers explicit (`A0..A3`, `C0..C3`, `P0..P3`)
- Handles boundary/partial tiles with zero-padding and `actual_extent`

### Dispatch skeleton

```python
@df.region()
def top(A0: Ty[IN], A1: Ty[IN], A2: Ty[IN], A3: Ty[IN],
        C0: Ty[OUT], C1: Ty[OUT], C2: Ty[OUT], C3: Ty[OUT],
        P0: Ty[PAR], P1: Ty[PAR], P2: Ty[PAR], P3: Ty[PAR]):
    @df.kernel(mapping=[4], args=[A0, A1, A2, A3, C0, C1, C2, C3, P0, P1, P2, P3])
    def core(lA0: Ty[IN] @ LyRep, lA1: Ty[IN] @ LyRep, lA2: Ty[IN] @ LyRep, lA3: Ty[IN] @ LyRep,
             lC0: Ty[OUT] @ LyRep, lC1: Ty[OUT] @ LyRep, lC2: Ty[OUT] @ LyRep, lC3: Ty[OUT] @ LyRep,
             lP0: Ty[PAR] @ LyRep, lP1: Ty[PAR] @ LyRep, lP2: Ty[PAR] @ LyRep, lP3: Ty[PAR] @ LyRep):
        pid, = df.get_pid()
        with allo.meta_if(pid == 0):
            kernel(lA0, lC0, lP0)
        with allo.meta_elif(pid == 1):
            kernel(lA1, lC1, lP1)
        with allo.meta_elif(pid == 2):
            kernel(lA2, lC2, lP2)
        with allo.meta_else():
            kernel(lA3, lC3, lP3)
```

---

## 3) What to Avoid

- Avoid writing examples that do not map to an existing `allo` or `mlir-aie` pattern.
- Avoid introducing standalone scalar-only kernel variants.
- Avoid introducing new project structure in generated tests.
- Avoid hard-coded assumptions about tile counts without memory-budget checks.

- Mapping/tiling logic is consistent across `.cc` and `_test.py`.