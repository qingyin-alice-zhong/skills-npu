// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv1d_bfloat16(bfloat16 in_buffer[256], bfloat16 out_buffer[254], std::int32_t param[3]) {
    event0();
    
    // Unpack parameters: kernel[0], kernel[1], stride (kernel values stored as float32 in int32)
    bfloat16 kernel[2];
    kernel[0] = (bfloat16)(*(float*)&param[0]);
    kernel[1] = (bfloat16)(*(float*)&param[1]);
    std::int32_t stride = param[2];
    
    constexpr std::int32_t VECTOR_SIZE = 256;
    constexpr std::int32_t KERNEL_SIZE = 2;
    std::int32_t num_windows = 254;  // Fixed output size

    for (std::int32_t i = 0; i < num_windows; i++) {
        float acc = 0.0f;
        for (std::int32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += (float)in_buffer[i * stride + j] * (float)kernel[j];
        }
        out_buffer[i] = (bfloat16)acc;
    }
    
    event1();
}

} // extern "C"