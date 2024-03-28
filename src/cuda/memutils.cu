// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <src/cuda/memutils.h>
#include <src/equihash/equihash.h>
#include <blake2b.h>

using namespace std;

// Allocate device memory
void allocateDeviceMemory(void** devPtr, size_t size)
{
    cudaMalloc(devPtr, size);
}

// Free device memory
void freeDeviceMemory(void* devPtr)
{
    cudaFree(devPtr);
}

// Copy data from host to device
void copyToDevice(void* dst, const void* src, const size_t size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}

// Copy data from device to host
void copyToHost(void* dst, const void* src, const size_t size)
{
    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}

// Launch CUDA kernel
//void launchKernel(void (*kernel)(/* params */), dim3 gridDim, dim3 blockDim /* params */) {
//    kernel<<<gridDim, blockDim>>>(/* params */);
//}

void CudaDeleter::operator()(void* ptr) const
{
    cudaFree(ptr);
}

// Helper function to create a unique_ptr with CUDA memory
template <typename T>
unique_ptr<T, CudaDeleter> make_cuda_unique(const size_t numElements)
{
    T* ptr = nullptr;
    cudaMalloc(&ptr, numElements * sizeof(T));
    return unique_ptr<T, CudaDeleter>(ptr);
}

template std::unique_ptr<uint32_t, CudaDeleter> make_cuda_unique<uint32_t>(const size_t numElements);
template std::unique_ptr<blake2b_state, CudaDeleter> make_cuda_unique<blake2b_state>(const size_t numElements);
template std::unique_ptr<Eh200_9::solution, CudaDeleter> make_cuda_unique<Eh200_9::solution>(const size_t numElements);