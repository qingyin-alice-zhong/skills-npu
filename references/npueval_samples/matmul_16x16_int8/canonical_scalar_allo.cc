// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void matmul_16x16_int8(std::int8_t a[256], std::int8_t b[256], std::int8_t out[256]) {
    // Matrix multiplication: out = a x b
    // Both a and b are 16x16 std::int8_t matrices in row-major order.
    // out is also a 16x16 std::int8_t matrix in row-major order.
    event0();
    constexpr int SIZE = 16;
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            std::int32_t acc = 0;
            for (int k = 0; k < SIZE; ++k) {
                acc += static_cast<std::int32_t>(a[i*SIZE + k]) * static_cast<std::int32_t>(b[k*SIZE + j]);
            }
            // Clamp to int8 range
            if (acc > 127) acc = 127;
            if (acc < -128) acc = -128;
            out[i*SIZE + j] = static_cast<std::int8_t>(acc);
        }
    }
    event1();
}

} // extern "C"