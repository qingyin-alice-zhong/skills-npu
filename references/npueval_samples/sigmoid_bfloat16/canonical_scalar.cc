// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

void sigmoid_bfloat16(bfloat16 *input_vector, bfloat16 *output_vector) {
    constexpr int32_t vector_size = 256;
    constexpr float exp_high = 88.3762626647949f;
    constexpr float exp_low = -87.3362626647949f;
    
    for (uint32_t i = 0; i < vector_size; i++) {
        float x = (float)input_vector[i];
        float result;

        if (x > exp_high) {
            result = 1.0f;
        }
        else if (x < exp_low) {
            result = 0.0f;
        }
        else {
            float t = -x;
            if (t > 0) {
                float sum = 1.0f + t;
                float term = t;
                for(int k = 2; k <= 6; k++) {
                    term *= (t / k);
                    sum += term;
                }
                result = 1.0f / (1.0f + sum);
            }
            else {
                t = -t;
                float sum = 1.0f + t;
                float term = t;
                for(int k = 2; k <= 6; k++) {
                    term *= (t / k);
                    sum += term;
                }
                result = sum / (1.0f + sum);
            }
        }
        output_vector[i] = (bfloat16)result;
    }
}