// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <blake2b.h>

#include <cuda_runtime.h>

__device__ bool blake2b_init_device(blake2b_state *S, size_t outlen);
__device__ void blake2b_update_device(blake2b_state *S, const void *pin, size_t inlen);
__device__ bool blake2b_final_device(blake2b_state *S, void *out, size_t outlen);
