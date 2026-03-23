// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void conv1d_bfloat16(bfloat16* in_buffer, bfloat16* kernel, bfloat16* out_buffer, uint32_t stride) {
    constexpr int32_t VECTOR_SIZE = 256;
    constexpr int32_t KERNEL_SIZE = 2;
    uint32_t num_windows = (VECTOR_SIZE - KERNEL_SIZE) / stride + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < KERNEL_SIZE; j++) {
            acc += (float)in_buffer[i * stride + j] * (float)kernel[j];
        }
        out_buffer[i] = (bfloat16)acc;
    }
}