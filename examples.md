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

### Vectorized kernel skeleton (adapted from `norm.cc`/`attn_out.cc`)

```cpp
#include <aie_api/aie.hpp>

template <typename T>
void kernel_vec(T *__restrict in, T *__restrict out, const int N) {
  T *__restrict in_base = in;
  T *__restrict out_base = out;
  constexpr int vec_factor = 256 / (sizeof(T) * 8);
  const int F = N / vec_factor;

  event0();
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; ++i) {
    aie::vector<T, vec_factor> v = aie::load_v<vec_factor>(in);
    in += vec_factor;

    // TODO: replace with op-specific vector math
    // e.g. aie::mul / aie::add / aie::sub / aie::max / reduce_add
    aie::vector<T, vec_factor> r = v;

    aie::store_v(out, r);
    out += vec_factor;
  }

  // tail handling for N % vec_factor != 0
  for (int i = F * vec_factor; i < N; ++i) {
    out_base[i] = in_base[i];
  }
  event1();
}
```

Use this only as a shape. The operation body must be copied/adapted from a real nearest neighbor kernel.

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

## 3) Vector Programming Rules (Imported from mlir-aie Guide)

Reference:
- `https://github.com/Xilinx/mlir-aie/tree/main/programming_guide`
- `section-4/section-4c/README.md`
- `quick_reference.md`

### Hard rules

1. Prefer vectorized kernels by default.
2. Derive `vec_factor` from dtype (`256/(sizeof(T)*8)` for 256b path or architecture-specific width in existing kernel family).
3. Use `__restrict` pointers for hot buffers.
4. Add both:
   - `AIE_PREPARE_FOR_PIPELINING`
   - `AIE_LOOP_MIN_ITERATION_COUNT(16)` (or operation-specific lower bound)
5. Use `aie::load_v` / `aie::store_v` and vector ops (`aie::mul`, `aie::add`, `aie::sub`, etc.) inside the innermost loop.
6. Keep tail handling for non-multiple lengths.
7. Validate with trace (`event0/event1`) and compare optimized variants when possible.

### Performance interpretation (from section-4c)

- Vectorization improves compute throughput, but load/store and DMA can still bottleneck.
- Always examine compute cycles and data movement cycles together.
- If VMAC utilization is low, inspect loop structure and scheduling pragmas first.

---

## 4) What to Avoid

- Avoid writing examples that do not map to an existing `allo` or `mlir-aie` pattern.
- Avoid introducing standalone scalar-only kernel variants.
- Avoid introducing new project structure in generated tests.
- Avoid hard-coded assumptions about tile counts without memory-budget checks.

---

## 5) Minimal checklist before shipping generated code

- Reference file paths used are listed in the response.
- Kernel and test signatures match (`input_idx`/`output_idx`, dtype, buffer sizes).
- Vectorized path exists (unless explicitly impossible).
- Tail handling is correct.
- Numeric tolerance matches dtype (`bfloat16` compares in float32).
- Mapping/tiling logic is consistent across `.cc` and `_test.py`.
