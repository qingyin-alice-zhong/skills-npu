# AMD AIEngine (AIE) API Documentation

## Overview

AIE API is a portable programming interface for AIE accelerators. It is implemented as a C++ header-only library that provides types and operations that get translated into efficient low-level intrinsics. The API also provides higher-level abstractions such as iterators and multi-dimensional arrays.

## Table of Contents

1. [Basic Types](#basic-types)
2. [Type Initialization](#type-initialization)
3. [Type Conversions](#type-conversions)
4. [Memory Operations](#memory-operations)
5. [Arithmetic Operations](#arithmetic-operations)
6. [Matrix Multiplication](#matrix-multiplication)
7. [Fast Fourier Transform (FFT)](#fast-fourier-transform-fft)
8. [Specialized Operations](#specialized-operations)

---

## Basic Types

The AIE API provides several fundamental types for vector operations:

### Vector Types

A vector represents a collection of elements of the same type which is transparently mapped to the corresponding vector registers supported on each architecture. Vectors are parametrized by the element type and the number of elements.

#### Supported Vector Types and Native Sizes

| Architecture | int4 | uint4 | int8 | uint8 | int16 | uint16 | int32 | uint32 | bfloat16 | float | cint16 | cint32 | cbfloat16 | cfloat |
|--------------|------|-------|------|-------|-------|--------|-------|--------|----------|-------|--------|--------|-----------|--------|
| AIE | - | - | 16/32/64/128 | 16/32/64/128 | 8/16/32/64 | - | 4/8/16/32 | - | - | 4/8/16/32 | 4/8/16/32 | 2/4/8/16 | - | 2/4/8/16 |
| AIE-ML/XDNA 1 | 32/64/128/256 | 32/64/128/256 | 16/32/64/128 | 16/32/64/128 | 8/16/32/64 | 8/16/32/64 | 4/8/16/32 | 4/8/16/32 | 8/16/32/64 | 4/8/16/32 | 4/8/16/32 | 2/4/8/16 | 4/8/16/32 | 2/4/8/16 |
| XDNA 2 | 32/64/128/256 | 32/64/128/256 | 16/32/64/128 | 16/32/64/128 | 8/16/32/64 | 8/16/32/64 | 4/8/16/32 | 4/8/16/32 | 8/16/32/64 | 4/8/16/32 | 4/8/16/32 | 2/4/8/16 | - | - |

#### Vector Declaration Example

```cpp
aie::vector<int16, 32> my_vector;
```

### Accumulator Types

An accumulator represents a collection of elements of the same class, typically obtained as a result of a multiplication operation. They provide a large amount of bits, allowing users to perform long chains of operations whose intermediate results would otherwise exceed the range of regular vector types.

#### Supported Accumulator Types and Native Sizes

| Architecture | acc32 | acc40 | acc48 | acc56 | acc64 | acc72 | acc80 | accfloat | cacc32 | cacc40 | cacc48 | cacc56 | cacc64 | cacc72 | cacc80 | caccfloat |
|--------------|-------|-------|-------|-------|-------|-------|-------|----------|--------|--------|--------|--------|--------|--------|--------|-----------|
| **AIE** |
| Lanes | 8/16/32/64/128 | 8/16/32/64/128 | 8/16/32/64/128 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16 |
| Native accumulation | 48b | 48b | 48b | 80b | 80b | 80b | 80b | 32b | 48b | 48b | 48b | 80b | 80b | 80b | 80b | 32b |
| **AIE-ML/XDNA 1** |
| Lanes | 8/16/32/64/128 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | - | - | 4/8/16/32/64/128 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | - | - | 2/4/8/16/32/64 |
| Native accumulation | 32b | 64b | 64b | 64b | 64b | - | - | 32b | 64b | 64b | 64b | 64b | 64b | - | - | 32b |
| **XDNA 2** |
| Lanes | 8/16/32/64/128 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | 4/8/16/32/64 | - | - | 4/8/16/32/64/128 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | 2/4/8/16/32 | - | - | - |
| Native accumulation | 32b | 64b | 64b | 64b | 64b | - | - | 32b | 64b | 64b | 64b | 64b | 64b | - | - | - |

#### Accumulator Declaration Example

```cpp
aie::accum<accfloat, 16> my_accumulator;
```

### Block Vector Types

A block vector represents a collection of blocked data types where several elements share some common data. Currently supported on XDNA 2 architecture.

#### Supported Block Vector Types

| Architecture | bfp16ebs8 | bfp16ebs16 |
|--------------|-----------|------------|
| XDNA 2 | 32/64/128/256 | 32/64/128/256 |

#### Block Vector Declaration Example

```cpp
aie::block_vector<bfp16ebs8, 64> my_vector;
```

### Mask Types

Masks are collections of values that can be 0 or 1, typically returned by comparison operations.

```cpp
aie::mask<64> my_mask;
```

### Accumulator Element Type Tags

The following tags are used to specify accumulator types:

- `acc32`, `acc40`, `acc48`, `acc56`, `acc64`, `acc72`, `acc80`: Integer accumulators with minimum bit requirements
- `accfloat`: Single precision floating point accumulator
- `cacc32`, `cacc40`, `cacc48`, `cacc56`, `cacc64`, `cacc72`, `cacc80`: Complex integer accumulators
- `caccfloat`: Complex single precision floating point accumulator

---

## Type Initialization

### Vector Initialization

#### Default Construction
```cpp
aie::vector<int16, 16> v; // Contents are undefined
```

#### Copy Construction
```cpp
aie::vector<int16, 16> v1;
aie::vector<int16, 16> v2 = v1;
```

#### From Operations
```cpp
aie::vector<int16, 16> v = aie::add(v1, v2);
```

#### From Memory
```cpp
aie::vector<int16, 16> v = aie::load_v<16>(ptr);
```

#### Element-wise Assignment
```cpp
aie::vector<int16, 16> v;
for (unsigned i = 0; i < v.size(); ++i)
    v[i] = i;
```

#### Subvector Assignment
```cpp
aie::vector<int16, 8> v1, v2;
aie::vector<int16, 16> v;
v.insert(0, v1); // Updates elements 0-7
v.insert(1, v2); // Updates elements 8-15
```

#### Vector Concatenation
```cpp
aie::vector<int16, 8> v1, v2;
aie::vector<int16, 16> v = aie::concat(v1, v2);
```

### Accumulator Initialization

#### Auto Type Deduction
```cpp
aie::vector<int16, 16> a = ...;
aie::vector<int32, 16> b = ...;
aie::accum result = aie::mul(a, b); // Type automatically deduced
```

#### Accessing Template Parameters
```cpp
using result_tag = decltype(result)::value_type;
constexpr size_t result_elems = result.size();
```

#### From Stream
```cpp
aie::vector<int16, 8> v = readincr_v<8>(input_stream);
aie::accum<acc48, 8> acc = readincr_v<8>(input_cascade);
```

### Mask Initialization

#### From Comparison
```cpp
aie::vector<int16, 16> a, b;
aie::mask<16> m = aie::lt(a, b);
```

#### From Constants
```cpp
auto m1 = aie::mask<64>::from_uint64(0xaaaabbbbccccddddULL);
auto m2 = aie::mask<64>::from_uint32(0xaaaabbbb, 0xccccdddd);
auto m3 = aie::mask<16>::from_uint32(0b1010'1010'1011'1011);
```

---

## Type Conversions

### Vector Casting

Vectors can be reinterpreted as vectors with different element types, as long as they have the same total size.

```cpp
aie::vector<int16, 32> v;
aie::vector<int32, 16> v2;
aie::vector<cint16, 16> v3;

v2 = v.cast_to<int32>();
v3 = aie::vector_cast<cint16>(v);
```

### Vector to Accumulator Conversion

```cpp
aie::vector<int16, 16> v;
aie::accum<acc32, 16> acc;

acc.from_vector(v, shift); // shift for fixed-point precision
```

### Accumulator to Vector Conversion

```cpp
aie::accum<acc32, 16> acc;
aie::vector<int16, 16> v;

v = acc.to_vector<int16>(shift); // shift before rounding and saturation
```

---

## Memory Operations

### Memory Alignment Requirements

| Architecture | 128b access | 256b access | 512b access |
|--------------|-------------|-------------|-------------|
| AIE | 128b | - | - |
| AIE-ML/XDNA 1 | 128b | 256b | - |
| XDNA 2 | 128b | 256b | 512b |

### Aligned Memory Access

```cpp
template <typename T>
T* aligned_memcpy(T* __restrict dest, const T* src, unsigned n)
{
    static constexpr unsigned Bits = 256;
    static constexpr unsigned Lanes = Bits / type_bits_v<T>;
    T* p = dest;
    
    for (unsigned i = 0; i < n / Lanes; ++i) {
        aie::vector<T, Lanes> v = aie::load_v<Lanes>(src);
        aie::store_v(p, v);
        src += Lanes;
        p += Lanes;
    }
    return dest;
}
```

### Unaligned Memory Access

```cpp
template <typename T>
T* unaligned_memcpy(T* __restrict dest, const T* src, unsigned n, 
                    unsigned dest_align = 1, unsigned src_align = 1)
{
    static constexpr unsigned Bits = 256;
    static constexpr unsigned Lanes = Bits / type_bits_v<T>;
    T* p = dest;
    
    for (unsigned i = 0; i < n / Lanes; ++i) {
        aie::vector<T, Lanes> v = aie::load_unaligned_v<Lanes>(src, src_align);
        aie::store_unaligned_v(p, v, dest_align);
        src += Lanes;
        p += Lanes;
    }
    return dest;
}
```

### Buffer Alignment

```cpp
alignas(aie::vector_decl_align) static int16 my_buffer[BUFFER_COUNT];
```

### Floor Load Operations

```cpp
alignas(aie::vector_decl_align) static int16 data[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

int16 *ptr = &data[3];
aie::vector<int16, 16> v = aie::load_floor_v(ptr, 16);
// Loads from aligned boundary: 0, 1, 2, ..., 15
```

### Iterator Types

| Iterator Type | Constructor | Iterator Kind |
|---------------|-------------|---------------|
| **Scalar Iterators** |
| Basic | `aie::begin` | Forward |
| Circular | `aie::begin_circular` | Forward |
| Random Circular | `aie::begin_random_circular` | Random Access |
| Pattern | `aie::begin_pattern` | Forward |
| **Vector Iterators** |
| Basic | `aie::begin_vector` | Random Access |
| Circular | `aie::begin_vector_circular` | Forward |
| Random Circular | `aie::begin_vector_random_circular` | Random Access |
| Restrict | `aie::begin_restrict_vector` | Random Access |
| Unaligned | `aie::begin_unaligned_vector` | Forward |

---

## Arithmetic Operations

### Default Accumulator Types for Multiplication

#### Integer Factor Types

| Type1 | Type2 | AIE | AIE-ML/XDNA 1 | XDNA 2 |
|-------|-------|-----|---------------|--------|
| int4 | int8 | - | acc32 | acc32 |
| int8 | int8 | acc48 | acc32 | acc32 |
| int8 | int16 | acc48 | acc32 | acc32 |
| int16 | int16 | acc48 | acc32 | acc32 |
| int16 | int32 | acc48 | acc64 | acc64 |
| int32 | int32 | acc80 | acc64¹ | acc64¹ |
| cint16 | int16 | cacc48 | cacc64 | cacc64 |
| cint16 | int32 | cacc48 | cacc64 | cacc64 |
| cint16 | cint16 | cacc48 | cacc64 | cacc64 |
| cint16 | cint32 | cacc48 | cacc64 | cacc64 |
| cint32 | int16 | cacc48 | cacc64 | cacc64 |
| cint32 | int32 | cacc80 | cacc64¹ | cacc64¹ |
| cint32 | cint32 | cacc80 | cacc64¹ | cacc64¹ |

¹ 32b x 32b multiplication is emulated using two 32b x 16b multiplications

#### Floating Point Factor Types

| Type1 | Type2 | AIE | AIE-ML/XDNA 1 | XDNA 2 |
|-------|-------|-----|---------------|--------|
| bfloat16 | bfloat16 | - | accfloat | accfloat |
| bfloat16 | cbfloat16 | - | caccfloat | caccfloat |
| cbfloat16 | cbfloat16 | - | caccfloat | caccfloat |
| float | float | accfloat | accfloat¹ | accfloat¹ |
| float | cfloat | caccfloat | caccfloat¹ | caccfloat¹ |
| cfloat | cfloat | caccfloat | caccfloat¹ | caccfloat¹ |
| bfp16ebs8 | bfp16ebs8 | - | - | accfloat |
| bfp16ebs16 | bfp16ebs16 | - | - | accfloat |

¹ Float multiplication is emulated using native bfloat16 multiplications

### Basic Arithmetic Example

```cpp
void add(int32 * __restrict out,
         const int32 * __restrict in1, 
         const int32 * __restrict in2, 
         unsigned count)
{
    for (unsigned i = 0; i < count; i += 8) {
        aie::vector<int32, 8> vec = aie::add(aie::load_v<8>(in1 + i),
                                             aie::load_v<8>(in2 + i));
        aie::store_v(out + i, vec);
    }
}
```

### Accumulator Type Selection

```cpp
// Default accumulation will be used
auto acc = aie::mul(v1, v2);

// 64b accumulation, at least, will be used
auto acc = aie::mul<acc64>(v1, v2);

// For multiply-add operations, API uses same accumulation as given accumulator
auto acc2 = aie::mac(acc, v1, v2);
```

---

## Matrix Multiplication

The AIE API encapsulates matrix multiplication functionality in the `aie::mmul` class template, parametrized with the matrix multiplication shape (MxKxN), data types, and optionally the accumulation precision.

### Basic MMUL Example

```cpp
template <unsigned M, unsigned K, unsigned N>
void mmul_blocked(unsigned rowA, unsigned colA, unsigned colB,
                  const int16 * __restrict pA, 
                  const int16 * __restrict pB, 
                  int16 * __restrict pC)
{
    using MMUL = aie::mmul<M, K, N, int16, int16>;

    for (unsigned z = 0; z < rowA; z += 2) {
        for (unsigned j = 0; j < colB; j += 2) {
            const int16 * __restrict pA1 = pA + (z * colA) * MMUL::size_A;
            const int16 * __restrict pA2 = pA + ((z + 1) * colA) * MMUL::size_A;
            const int16 * __restrict pB1 = pB + (j) * MMUL::size_B;
            const int16 * __restrict pB2 = pB + (j + 1) * MMUL::size_B;

            aie::vector<int16, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
            aie::vector<int16, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2);
            aie::vector<int16, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
            aie::vector<int16, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2);

            MMUL C00; C00.mul(A0, B0);
            MMUL C01; C01.mul(A0, B1);
            MMUL C10; C10.mul(A1, B0);
            MMUL C11; C11.mul(A1, B1);

            for (unsigned i = 1; i < colA; ++i) {
                A0 = aie::load_v<MMUL::size_A>(pA1 + i * MMUL::size_A);
                A1 = aie::load_v<MMUL::size_A>(pA2 + i * MMUL::size_A);
                B0 = aie::load_v<MMUL::size_B>(pB1 + i * MMUL::size_B * colB);
                B1 = aie::load_v<MMUL::size_B>(pB2 + i * MMUL::size_B * colB);

                C00.mac(A0, B0);
                C01.mac(A0, B1);
                C10.mac(A1, B0);
                C11.mac(A1, B1);
            }

            int16 * __restrict pC1 = pC + (z * colB + j) * MMUL::size_C;
            int16 * __restrict pC2 = pC + ((z + 1) * colB + j) * MMUL::size_C;
            
            aie::store_v(pC1, C00.template to_vector<int16>());
            aie::store_v(pC1 + MMUL::size_C, C01.template to_vector<int16>());
            aie::store_v(pC2, C10.template to_vector<int16>());
            aie::store_v(pC2 + MMUL::size_C, C11.template to_vector<int16>());
        }
    }
}
```

---

## Fast Fourier Transform (FFT)

The AIE API offers a stage-based interface for carrying out decimation-in-time FFTs.

### FFT Example (1024-point)

```cpp
void fft_1024pt(const cint16 * __restrict x,
                unsigned shift_tw,    // Twiddle decimal point
                unsigned shift,       // Output shift
                bool inv,            // Inverse FFT flag
                cint16 * __restrict tmp,
                cint16 * __restrict y)
{
    aie::fft_dit_r2_stage<512>(x,   tw1,   1024, shift_tw, shift, inv, tmp);
    aie::fft_dit_r2_stage<256>(tmp, tw2,   1024, shift_tw, shift, inv, y);
    aie::fft_dit_r2_stage<128>(y,   tw4,   1024, shift_tw, shift, inv, tmp);
    aie::fft_dit_r2_stage<64> (tmp, tw8,   1024, shift_tw, shift, inv, y);
    aie::fft_dit_r2_stage<32> (y,   tw16,  1024, shift_tw, shift, inv, tmp);
    aie::fft_dit_r2_stage<16> (tmp, tw32,  1024, shift_tw, shift, inv, y);
    aie::fft_dit_r2_stage<8>  (y,   tw64,  1024, shift_tw, shift, inv, tmp);
    aie::fft_dit_r2_stage<4>  (tmp, tw128, 1024, shift_tw, shift, inv, y);
    aie::fft_dit_r2_stage<2>  (y,   tw256, 1024, shift_tw, shift, inv, tmp);
    aie::fft_dit_r2_stage<1>  (tmp, tw512, 1024, shift_tw, shift, inv, y);
}
```

### Mixed Radix FFT Example (512-point)

```cpp
void fft_512pt(const cint16 * __restrict x,
               unsigned shift_tw, unsigned shift, bool inv,
               cint16 * __restrict tmp, cint16 * __restrict y)
{
    aie::fft_dit_r2_stage<256>(x,   tw1,                     512, shift_tw, shift, inv, y);
    aie::fft_dit_r4_stage<64> (y,   tw2,   tw4,   tw2_4,     512, shift_tw, shift, inv, tmp);
    aie::fft_dit_r4_stage<16> (tmp, tw8,   tw16,  tw8_16,    512, shift_tw, shift, inv, y);
    aie::fft_dit_r4_stage<4>  (y,   tw32,  tw64,  tw32_64,   512, shift_tw, shift, inv, tmp);
    aie::fft_dit_r4_stage<1>  (tmp, tw128, tw256, tw128_256, 512, shift_tw, shift, inv, y);
}
```

### Twiddle Generation

For an R-Radix, N-point FFT, twiddles can be computed as:

```cpp
int n_stage = N / Vectorization;
int n_tws = n_stage / Radix;
for (unsigned r = 1; r < Radix; ++r) {
    for (unsigned i = 0; i < n_tws; ++i) {
        tw[r-1][i] = exp(-2j * pi * r * i / n_stage);
    }
}
```

---

## Specialized Operations

### Floating-Point Conversion Support

#### Float to Fixed Conversion Support

| Output Bits | Type | Architecture | bfloat16 | float |
|-------------|------|--------------|----------|-------|
| 4b | Scalar | AIE-ML/XDNA 1 | Vector unit | Vector unit |
| 4b | Vector | AIE-ML/XDNA 1 | Emulated (symmetric_zero) | Emulated (symmetric_zero) |
| 8b | Scalar | AIE-ML/XDNA 1 | Vector unit | Vector unit |
| 8b | Vector | AIE-ML/XDNA 1 | Emulated (symmetric_zero) | Emulated (symmetric_zero) |
| 16b | Scalar | AIE-ML/XDNA 1 | Vector unit | Vector unit |
| 16b | Vector | AIE-ML/XDNA 1 | Native to 32b + extract (floor) | Emulated (symmetric_zero) |
| 32b | Scalar | AIE-ML/XDNA 1 | Vector unit | Vector unit |
| 32b | Vector | AIE-ML/XDNA 1 | Native (floor) | Emulated (symmetric_zero) |

### Lookup Tables (AIE-ML/XDNA 1+)

#### Parallel Lookup

```cpp
template <typename Value>
void parallel_lookup(const int8* pIn, Value* pOut, 
                     const aie::lut<4, Value>& my_lut,
                     int samples, int step_bits, int bias, int LUT_elems)
{
    aie::parallel_lookup<int8, aie::lut<4, Value>> lookup(my_lut, step_bits, bias);

    auto it_in = aie::begin_vector<32>(pIn);
    auto it_out = aie::begin_vector<32>(pOut);

    for (unsigned l = 0; l < samples / 32; ++l)
        *it_out++ = lookup.fetch(*it_in++);
}
```

#### Linear Approximation

```cpp
template <typename OffsetType, typename SlopeType>
void linear_approx(const int8* pIn, OffsetType* pOut, 
                   const aie::lut<4, OffsetType, SlopeType>& my_lut,
                   int samples, int step_bits, int bias, int LUT_elems, 
                   int shift_offset, int shift_out)
{
    aie::linear_approx<int8, aie::lut<4, OffsetType, SlopeType>> lin_approx(my_lut, step_bits, bias, shift_offset);

    auto it_in = aie::begin_vector<32>(pIn);
    auto it_out = aie::begin_vector<32>(pOut);

    for (unsigned l = 0; l < samples / 32; ++l)
        *it_out++ = lin_approx.compute(*it_in++).to_vector<OffsetType>(shift_out);
}
```

### Sparse Vector Operations

#### Sparse Vector Input Buffer Streams

```cpp
auto vbs = aie::sparse_vector_input_buffer_stream<int8, 128>(ptr);

aie::sparse_vector<int8, 128> a, b, c;
vbs >> a;           // Single read
vbs >> b >> c;      // Multiple reads
auto d = vbs.pop(); // Type deduced automatically
```

#### Sparse Data Format

Sparse data requires minimum 50% sparsity (two zero values within each group of four consecutive values). The layout consists of:
- 64-bit mask (aligned to 32b boundary)
- Compressed data following the mask

**Sparse Partial Decompression Table:**

| Mask bits (4) | Partially decompressed data (2) |
|---------------|----------------------------------|
| 0 0 0 0 | 0, 0 |
| 0 0 0 1 | 0, A |
| 0 0 1 0 | 0, B |
| 0 0 1 1 | B, A |
| 0 1 0 0 | C, 0 |
| 0 1 0 1 | C, A |
| 0 1 1 0 | C, B |
| 1 0 0 0 | D, 0 |
| 1 0 0 1 | D, A |
| 1 0 1 0 | D, B |
| 1 1 0 0 | D, C |

### Block Vector Buffer Streams

Block vector types exceed core memory interface sizes and require special FIFO interfaces:

```cpp
T *ptr; // T is a block type, such as bfp16ebs8

aie::vector<bfloat16, 64> data(...);
aie::accum<accfloat, 64> acc(data);

// Write block vector to memory
aie::block_vector_output_buffer_stream<T, 64> out_stream(ptr);
out_stream << acc.to_vector<T>();

// Read block vector from memory
aie::block_vector<T, 64> v;
aie::block_vector_input_buffer_stream<T, 64> in_stream(ptr);
in_stream >> v;
```

### Tensor Buffer Streams (AIE-ML/XDNA 1+)

Multi-dimensional addressing abstraction for tensor operations:

#### Tensor Descriptor Creation

```cpp
auto desc = aie::make_tensor_descriptor<int16, 32>(
    aie::tensor_dim(2u, 4),   // size, step
    aie::tensor_dim(2u, 2),
    aie::tensor_dim(2u, 1));

auto tbs = aie::make_tensor_buffer_stream(ptr, desc);

for (unsigned i = 0; i < 8; ++i) {
    aie::vector<int16, 32> v;
    tbs >> v;
    // Process vector...
}
```

#### Native Tensor Descriptors

```cpp
// Manual decomposition using native types
aie::make_tensor_descriptor_from_native<int16, 32>(
    aie::dim_3d(1u, 1,   // num1, inc1
                1u, 1,   // num2, inc2
                    1)); //       inc3
```

#### Nested Tensor Streams

For dimensions > 3, tensors decompose recursively:

```cpp
auto desc = aie::make_tensor_descriptor<int16, 32>(
    aie::tensor_dim(2u, 4),
    aie::tensor_dim(2u, 0),   // step = 0 for iteration
    aie::tensor_dim(6u, 8),
    aie::tensor_dim(4u, 1));

auto tbs = aie::make_tensor_buffer_stream(ptr, desc);

for (unsigned i = 0; i < 2*2*6; ++i) {
    auto tbs_inner = tbs.pop(); // Get inner stream
    
    aie::vector<int16, 32> a, b, c, d;
    tbs_inner >> a >> b >> c >> d;
}
```

---

## Operator Overloading

The AIE API provides operator overloading for intuitive syntax:

```cpp
#include <aie_api/operators.hpp>

using namespace aie::operators;

aie::mask<16> less_than_add(aie::vector<int32, 16> a, 
                           aie::vector<int32, 16> b, 
                           aie::vector<int32, 16> c)
{
    return c < (a + b);
}
```

---

## ADF Graph Interoperability

AIE API extends ADF abstractions to work with vector and accumulator types:

```cpp
aie::vector<int16, 8> v = readincr_v<8>(input_stream);
aie::accum<acc48, 8> acc = readincr_v<8>(input_cascade);
```

**Note:** ADF accumulator abstractions require natively supported accumulator tags on the target architecture.

---

## Memory Bank Conflicts and Virtual Resources

### Type Annotations

Prevent bank conflicts using virtual resource annotations:

```cpp
void fn(int __aie_dm_resource_a * A,
        int                     * B,
        int __aie_dm_resource_a * C)
{
    aie::vector<int, 8> v1 = aie::load_v<8>(A);  // Bound to resource 'a'
    aie::vector<int, 8> v2 = aie::load_v<8>(B);  // No annotation
    aie::vector<int, 8> v3 = aie::load_v<8>(C);  // Bound to resource 'a'
}
```

### Function-level Resource Binding

```cpp
void fn(int __aie_dm_resource_a * A, int * B)
{
    aie::vector<int, 8> v1 = aie::load_v<8>(A);
    aie::vector<int, 8> v2 = aie::load_v<8>(B);
    // This specific access to B uses resource 'a'
    aie::vector<int, 8> v3 = aie::load_v<8, aie_dm_resource::a>(B);
}
```

---

## Lazy Operations

AIE API provides wrapper types for input arguments to merge operations:

```cpp
aie::accum<cacc48, 16> foo(aie::vector<int16, 16> a, 
                          aie::vector<cint16, 16> b)
{
    aie::accum<cacc48, 16> ret;
    
    // Performs element-wise multiplication of abs(a) and conj(b)
    ret = aie::mul(aie::op_abs(a), aie::op_conj(b));
    
    return ret;
}
```

---

## Best Practices and Notes

### Important Warnings

1. **Undefined Vectors:** Operations with undefined input vectors may not produce errors during compilation but may not work as expected.

2. **Undefined Accumulators:** Reading undefined accumulators may not produce compilation errors but operations may not behave as intended.

3. **Non-native Accumulators:** ADF graph code does not support non-native accumulator types.

4. **Alignment Requirements:** Always ensure proper memory alignment for optimal performance.

### Performance Considerations

1. **Bank Conflicts:** Use virtual resource annotations to prevent memory bank conflicts.

2. **Unaligned Access Overhead:** Unaligned memory accesses may incur additional overhead depending on misalignment amount.

3. **Vector Size Selection:** Choose vector sizes that match native hardware capabilities for optimal performance.

4. **Accumulator Precision:** Use appropriate accumulator precision to balance performance and accuracy requirements.

### Architecture-Specific Notes

- **AIE:** Original architecture with 48b/80b native accumulation
- **AIE-ML/XDNA 1:** Enhanced with ML capabilities, 32b/64b native accumulation, bfloat16 support
- **XDNA 2:** Latest architecture with block vector support, enhanced floating-point capabilities

This comprehensive documentation should serve as an excellent reference for LLMs and coding agents working with the AMD AIEngine API. The structured format makes it easy to understand capabilities, supported types, and implementation patterns across different AIE architectures.