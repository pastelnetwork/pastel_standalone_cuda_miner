
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <src/cuda/memutils.h>
#include <src/cuda/blake2b_device.h>
#include <src/equihash/equihash.h>

using namespace std;

// Get the number of available CUDA devices
int getNumCudaDevices()
{
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    return numDevices;
}

// Get the maximum number of threads per block supported by the device
int getMaxThreadsPerBlock(int deviceId)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    return deviceProp.maxThreadsPerBlock;
}

// Calculate the grid and block dimensions based on the problem size and device capabilities
void calculateGridAndBlockDims(dim3& gridDim, dim3& blockDim, size_t nHashes, int nThreadsPerHash, int deviceId)
{
    int nMaxThreadsPerBlock = getMaxThreadsPerBlock(deviceId);
    int nBlocks = (nHashes + nThreadsPerHash - 1) / nThreadsPerHash;

    blockDim.x = min(nThreadsPerHash, nMaxThreadsPerBlock);
    gridDim.x = (nBlocks + blockDim.x - 1) / blockDim.x;
}

// CUDA kernel to generate initial hashes from blake2b state
template<typename EquihashType>
__global__ void cudaKernel_generateInitialHashes(const blake2b_state* state, uint32_t* hashes)
{
    const uint32_t hashIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (hashIdx < EquihashType::NHashes)
    {
        const uint32_t blockIdx = hashIdx / EquihashType::IndicesPerHashOutput;

        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&blockIdx), sizeof(blockIdx));

        uint8_t hash[EquihashType::HashOutput];  
        blake2b_final_device(&localState, hash, EquihashType::HashOutput);

        const uint32_t outputIdx = hashIdx * EquihashType::HashWords;
        for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
            hashes[outputIdx + i] = (reinterpret_cast<uint32_t*>(hash))[i];
    }
}

template<typename EquihashType>
void generateInitialHashes(const blake2b_state* devState, uint32_t* devHashes,
    const uint32_t threadsPerBlock)
{
    dim3 gridDim((EquihashType::NBlocks + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDim(threadsPerBlock);

    cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(devState, devHashes);
}

/**
 * Perform on-the-fly collision detection by distributing hashes into buckets based on their 
 * leading bits and storing their indices for further processing. Each thread processes one hash 
 * and determines its bucket based on the current round of the algorithm.
 * 
 * This kernel populates a bucket with the indices of hashes that belong to it, which are used in 
 * subsequent steps to detect and process collisions.
 * 
 * \tparam EquihashType A struct providing constants specific to the Equihash configuration.
 * \param hashes Array of hash values.
 * \param indices Array where the indices of hashes will be stored, organized by buckets.
 * \param bucketSizes Array that tracks the number of indices stored in each bucket.
 * \param round The current round number, affecting the hash portion used for bucketing.
 */
template <typename EquihashType>
__global__ void cudaKernel_detectCollisions(uint32_t* hashes, uint32_t* slotBitmaps)
{
    const uint32_t hashIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (hashIdx < EquihashType::NHashes)
    {
        const uint32_t bucketIdx = (hashes[hashIdx * EquihashType::HashWords] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
        const uint32_t slotIdx = hashIdx % EquihashType::NSlots;
        const uint32_t bitmapIdx = slotIdx / 32;
        const uint32_t bitmapMask = 1U << (slotIdx % 32);

        atomicOr(&slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx], bitmapMask);
    }
}

template <typename EquihashType>
void detectCollisions(uint32_t* devHashes, uint32_t* devSlotBitmaps, const uint32_t threadsPerBlock)
{
    // Clear the slot bitmaps
    cudaMemset(devSlotBitmaps, 0, EquihashType::NSlots * (EquihashType::NSlots / 32) * sizeof(uint32_t));

    dim3 gridDim((EquihashType::NHashes + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDim(threadsPerBlock);

    cudaKernel_detectCollisions<EquihashType><<<gridDim, blockDim>>>(devHashes, devSlotBitmaps);
}

template<typename EquihashType>
__global__ void cudaKernel_xorCollisions(uint32_t* hashes, uint32_t* slotBitmaps, uint32_t* xoredHashes)
{
    const uint32_t bucketIdx = blockIdx.x;
    const uint32_t slotIdx = threadIdx.x;

    if (slotIdx >= EquihashType::NSlots)
        return;

    const uint32_t bitmapIdx = slotIdx / 32;
    const uint32_t bitmapMask = 1U << (slotIdx % 32);

    if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx] & bitmapMask)
    {
        const uint32_t index1 = bucketIdx * EquihashType::NSlots + slotIdx;
        const uint32_t index2 = bucketIdx * EquihashType::NSlots + ((slotIdx + 1) % EquihashType::NSlots);
        const uint32_t slotIdx = bucketIdx * (EquihashType::NSlots / 32) + index2 / 32;

        if (slotBitmaps[slotIdx] & (1U << (index2 % 32)))
        {
            for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
                xoredHashes[index1 * EquihashType::HashWords + i] ^= hashes[index2 * EquihashType::HashWords + i];
        }
    }
}

template<typename EquihashType>
void xorCollisions(uint32_t* devHashes, uint32_t* devSlotBitmaps, uint32_t* devXoredHashes, const uint32_t threadsPerBlock)
{
    const dim3 gridDim(EquihashType::NSlots);
    const dim3 blockDim(threadsPerBlock);

    cudaKernel_xorCollisions<EquihashType><<<gridDim, blockDim>>>(devHashes, devSlotBitmaps, devXoredHashes);
}

/**
 * @brief Find valid solutions by checking the XORed values against the target difficulty.
 * 
 * @param hashes - Array of hash values 
 * @param solutions - Array to store the valid solutions
 * @param solutionCount - The number of valid solutions found 
 */
template<typename EquihashType>
__global__ void cudaKernel_findSolutions(uint32_t* hashes, uint32_t* slotBitmaps, typename EquihashType::solution* solutions, uint32_t* solutionCount)
{
    const uint32_t hashIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (hashIdx >= EquihashType::NHashes)
        return;
    // Check if the hash meets the difficulty target
    if (hashes[hashIdx * EquihashType::HashWords] == 0)
    {
        // Calculate the bucket index and slot index
        const uint32_t bucketIdx = (hashes[hashIdx * EquihashType::HashWords] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
        const uint32_t slotIdx = hashIdx % EquihashType::NSlots;

        // Check if the slot is marked in the slotBitmap
        const uint32_t bitmapIdx = slotIdx / 32;
        const uint32_t bitmapMask = 1U << (slotIdx % 32);
        if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx] & bitmapMask)
        {
            // Atomically increment the solution count and get the current index
            const uint32_t solutionIdx = atomicAdd(solutionCount, 1);

            // Store the solution indices
            uint32_t indiceIdx = 0;
            uint32_t indiceValue = hashIdx;
            for (uint32_t i = 0; i < EquihashType::WK; ++i)
            {
                solutions[solutionIdx].indices[indiceIdx] = indiceValue;
                ++indiceIdx;

                // Calculate the next indice value
                const uint32_t parentIdx = indiceValue / 2;
                const uint32_t parentSlotIdx = parentIdx % EquihashType::NSlots;
                const uint32_t parentBitmapIdx = parentSlotIdx / 32;
                const uint32_t parentBitmapMask = 1U << (parentSlotIdx % 32);

                if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + parentBitmapIdx] & parentBitmapMask)
                    indiceValue = parentIdx;
                else
                    indiceValue = parentIdx + EquihashType::NSlots;
            }        
        }
    }
 }

template<typename EquihashType>
uint32_t findSolutions(uint32_t* devHashes, uint32_t* devSlotBitmaps, typename EquihashType::solution* devSolutions, uint32_t* devSolutionCount, 
    const uint32_t threadsPerBlock)
{
    // Clear the solution count
    cudaMemset(devSolutionCount, 0, sizeof(uint32_t));

    // Launch the kernel to find valid solutions
    const dim3 gridDim((EquihashType::NHashes + threadsPerBlock - 1) / threadsPerBlock);
    const dim3 blockDim(threadsPerBlock);
    cudaKernel_findSolutions<EquihashType><<<gridDim, blockDim>>>(devHashes, devSlotBitmaps, devSolutions, devSolutionCount);

    // Copy the solution count from device to host
    uint32_t solutionCount;
    copyToHost(&solutionCount, devSolutionCount, sizeof(uint32_t));

    return solutionCount;
}

template<typename EquihashType>
void copySolutionsToHost(typename EquihashType::solution* devSolutions, const uint32_t nSolutionCount, vector<typename EquihashType::solution> &vHostSolutions)
{
    vHostSolutions.clear();
    // Resize the host solutions vector
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), devSolutions, nSolutionCount * EquihashType::ProofSize);
}

// Explicit template instantiation
template void generateInitialHashes<Eh200_9>(const blake2b_state* devState, uint32_t* devHashes, const uint32_t threadsPerBlock);
template void detectCollisions<Eh200_9>(uint32_t* devHashes, uint32_t* devSlotBitmaps, const uint32_t threadsPerBlock);
template void xorCollisions<Eh200_9>(uint32_t* devHashes, uint32_t* devSlotBitmaps, uint32_t* devXoredHashes, const uint32_t threadsPerBlock);
template uint32_t findSolutions<Eh200_9>(uint32_t* devHashes, uint32_t* devSlotBitmaps, Eh200_9::solution* devSolutions, uint32_t* devSolutionCount, const uint32_t threadsPerBlock);
template void copySolutionsToHost<Eh200_9>(Eh200_9::solution* devSolutions, const uint32_t nSolutionCount, vector<Eh200_9::solution> &vHostSolutions);
