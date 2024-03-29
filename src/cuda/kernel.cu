
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
        const uint32_t blockIndex = hashIdx / EquihashType::IndicesPerHashOutput;

        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&blockIndex), sizeof(blockIndex));

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
 * \param slotBitmaps
 */
template <typename EquihashType>
__global__ void cudaKernel_detectCollisions(uint32_t* hashes, uint32_t* slotBitmaps)
{
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t hashIdx = gid * EquihashType::HashWords;

    if (hashIdx >= EquihashType::NHashes * EquihashType::HashWords)
        return;

    const uint32_t bucketIdx = (hashes[hashIdx] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
    const uint32_t slotIdx = hashIdx % EquihashType::NSlots;
    const uint32_t bitmapIdx = slotIdx / 32;
    const uint32_t bitmapMask = 1U << (slotIdx % 32);

    atomicOr(&slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx], bitmapMask);
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
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t hashIdx = gid * EquihashType::HashWords;

    if (hashIdx >= EquihashType::NHashes * EquihashType::HashWords)
        return;

    const uint32_t bucketIdx = (hashes[hashIdx] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
    const uint32_t slotIdx = hashIdx % EquihashType::NSlots;
    const uint32_t bitmapIdx = slotIdx / 32;
    const uint32_t bitmapMask = 1U << (slotIdx % 32);

    if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx] & bitmapMask)
    {
        const uint32_t index1 = hashIdx;
        uint32_t index2 = index1 ^ (1U << (EquihashType::CollisionBitLength - 1));

        while (index2 < EquihashType::NHashes * EquihashType::HashWords)
        {
            const uint32_t index2BitmapIdx = index2 / 32;
            const uint32_t index2BitmapMask = 1U << (index2 % 32);

            if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + index2BitmapIdx] & index2BitmapMask)
            {
                #pragma unroll
                for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
                    xoredHashes[index1 + i] ^= hashes[index2 + i];
            }
            index2 += (1U << (EquihashType::CollisionBitLength - 1));
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
    const uint32_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t hashIdx = gid * EquihashType::HashWords;

    if (hashIdx >= EquihashType::NHashes * EquihashType::HashWords)
        return;

    const uint32_t bucketIdx = (hashes[hashIdx] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
    const uint32_t slotIdx = hashIdx % EquihashType::NSlots;
    const uint32_t bitmapIdx = slotIdx / 32;
    const uint32_t bitmapMask = 1U << (slotIdx % 32);

    if (slotBitmaps[bucketIdx * (EquihashType::NSlots / 32) + bitmapIdx] & bitmapMask)
    {
        uint32_t solutionIndices[EquihashType::ProofSize];
        uint32_t xoredHash[EquihashType::HashWords] = { 0 };

        // Initialize the solution indices and xoredHash
        solutionIndices[0] = hashIdx / EquihashType::HashWords;
        for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
            xoredHash[i] = hashes[hashIdx + i];

        // Traverse the tree to find a valid solution
        for (uint32_t depth = 1; depth < EquihashType::ProofSize; ++depth)
        {
            const uint32_t parentIdx = solutionIndices[depth - 1];
            const uint32_t parentBucketIdx = (hashes[parentIdx * EquihashType::HashWords] >> (32 - EquihashType::CollisionBitLength)) & (EquihashType::NSlots - 1);
            const uint32_t parentSlotIdx = parentIdx % EquihashType::NSlots;

            uint32_t childIdx = parentIdx ^ (1U << (depth - 1));
            bool childFound = false;

            while (childIdx < EquihashType::NHashes)
            {
                const uint32_t childBitmapIdx = childIdx / 32;
                const uint32_t childBitmapMask = 1U << (childIdx % 32);

                if (slotBitmaps[parentBucketIdx * (EquihashType::NSlots / 32) + childBitmapIdx] & childBitmapMask)
                {
                    solutionIndices[depth] = childIdx;
                    for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
                        xoredHash[i] ^= hashes[childIdx * EquihashType::HashWords + i];
                    childFound = true;
                    break;
                }

                childIdx += (1U << (depth - 1));
            }

            if (!childFound)
                break;
        }

        // Check if the xoredHash satisfies the difficulty target
        // TODO: Implement the difficulty check based on the specific target

        // If a valid solution is found, store it
        if (true/* Difficulty check passed */)
        {
            const uint32_t solutionIdx = atomicAdd(solutionCount, 1);
            for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
                solutions[solutionIdx].indices[i] = solutionIndices[i];
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
    uint32_t solutionCount = 0;
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
