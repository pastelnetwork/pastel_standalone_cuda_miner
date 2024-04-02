// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

#include <blake2b.h>

struct GpuMemDeleter
{
    void operator()(void* ptr) const;
};

// Helper functions to create a unique_ptr with CUDA memory
template <typename T>
std::unique_ptr<T, GpuMemDeleter> make_gpu_unique(const size_t numElements);


void copyToDevice(void* dst, const void* src, const size_t size);
void copyToHost(void* dst, const void* src, const size_t size);

