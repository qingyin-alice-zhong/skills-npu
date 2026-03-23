// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void relu_int8(std::int8_t in_buffer[1024], std::int8_t out_buffer[1024]) {
    event0();
    constexpr std::int32_t num_elements = 1024;
    for (std::uint32_t i = 0; i < num_elements; i++) {
        std::int8_t v = in_buffer[i];
        out_buffer[i] = (v > 0) ? v : 0;
    }
    event1();
}

} // extern "C"