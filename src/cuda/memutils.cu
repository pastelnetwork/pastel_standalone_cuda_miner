// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <iostream>

#include <src/cuda/memutils.h>
#include <src/equihash/equihash.h>
#include <blake2b.h>

using namespace std;

#define CUDA_CHECK(call)                                                   \
do {                                                                       \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " in "   \
                  << __FILE__ << ":" << __LINE__ << std::endl;             \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
} while (0)

// Allocate device memory
void allocateDeviceMemory(void** devPtr, size_t size)
{
    CUDA_CHECK(cudaMalloc(devPtr, size));
}

// Free device memory
void freeDeviceMemory(void* devPtr)
{
    CUDA_CHECK(cudaFree(devPtr));
}

// Copy data from host to device
void copyToDevice(void* dst, const void* src, const size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

// Copy data from device to host
void copyToHost(void* dst, const void* src, const size_t size)
{
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void CudaDeleter::operator()(void* ptr) const
{
    cudaFree(ptr);
}

// Helper function to create a unique_ptr with CUDA memory
template <typename T>
unique_ptr<T, CudaDeleter> make_cuda_unique(const size_t numElements)
{
    T* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, numElements * sizeof(T)));
    return unique_ptr<T, CudaDeleter>(devPtr);
}

template <typename T>
unique_ptr<T, CudaDeleter> make_cuda_unique_2d(const size_t rowSize, const size_t columnSize, size_t &pitch)
{
    T* devPtr = nullptr;
    CUDA_CHECK(cudaMallocPitch(&devPtr, &pitch, rowSize * sizeof(T), columnSize));
    return unique_ptr<T, CudaDeleter>(devPtr);
}

template std::unique_ptr<uint32_t, CudaDeleter> make_cuda_unique<uint32_t>(const size_t numElements);
template std::unique_ptr<uint32_t*, CudaDeleter> make_cuda_unique<uint32_t*>(const size_t numElements);
template std::unique_ptr<blake2b_state, CudaDeleter> make_cuda_unique<blake2b_state>(const size_t numElements);
template std::unique_ptr<Eh200_9::solution_device_type, CudaDeleter> make_cuda_unique<Eh200_9::solution_device_type>(const size_t numElements);

template std::unique_ptr<uint32_t, CudaDeleter> make_cuda_unique_2d(const size_t rowSize, const size_t columnSize, size_t &pitch);