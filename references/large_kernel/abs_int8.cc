// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void abs_int8(std::int8_t in_buffer[1024], std::int8_t out_buffer[1024]) {
    event0();
    constexpr int vec_factor = 32; // Maintain increased vector factor for parallelism
    std::int8_t *__restrict pIn = in_buffer;
    std::int8_t *__restrict pOut = out_buffer;
    const int F = 1024 / vec_factor;
    for (int i = 0; i < F; i++)
        chess_prepare_for_pipelining chess_loop_range(32, ) {
            aie::vector<int8_t, vec_factor> inVec = aie::load_v<vec_factor>(pIn);
            pIn += vec_factor;
            aie::vector<int8_t, vec_factor> absVec = aie::abs(inVec);
            aie::store_v(pOut, absVec);
            pOut += vec_factor;
        }
    event1();
}

} // extern "C"