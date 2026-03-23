// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void relu_int8(int8_t *in_buffer, int8_t* out_buffer) {
    constexpr int32_t num_elements = 1024;
    for (uint32_t i = 0; i < num_elements; i++) {
        int8_t v = in_buffer[i];
        out_buffer[i] = (v > 0) ? v : 0;
    }
}