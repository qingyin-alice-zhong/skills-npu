// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void avgpool2d_bfloat16(bfloat16* input, bfloat16* output) {
    constexpr int32_t ROWS = 32;
    constexpr int32_t COLS = 32;
    constexpr int32_t WINDOW_SIZE = 2;
    constexpr int32_t STRIDE = 2;
    constexpr int32_t WINDOW_AREA = WINDOW_SIZE * WINDOW_SIZE;

    for (int i = 0; i < ROWS; i += STRIDE) {
        for (int j = 0; j < COLS; j += STRIDE) {
            float sum = 0.0f;
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
                    float current_val = (float)input[(i + wi) * COLS + (j + wj)];
                    sum += current_val;
                }
            }
            float avg = sum / WINDOW_AREA;
            output[(i/STRIDE) * (COLS/STRIDE) + (j/STRIDE)] = (bfloat16)avg;
        }
    }
}