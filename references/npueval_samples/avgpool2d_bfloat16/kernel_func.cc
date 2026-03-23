// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void avgpool2d_bfloat16(bfloat16 input[1024], bfloat16 output[256]) {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"