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
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"