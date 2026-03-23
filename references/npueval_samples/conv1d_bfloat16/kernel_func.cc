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
    // TODO: Implement the kernel
    event1();
}

} // extern "C"