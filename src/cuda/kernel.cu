
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <vector>
#include <string>
#include <bitset>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <src/cuda/kernel.h>
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

template<typename EquihashType>
bool EhDevice<EquihashType>::allocate_memory()
{
    try
    {
        // Allocate device memory for blake2b state
        initialState = make_cuda_unique<blake2b_state>(1);
        // Allocate device memory for initial hash values
        hashes = make_cuda_unique<uint32_t>(EquihashType::NHashWords);
        // Allocate device memory for XORed hash values
        xoredHashes = make_cuda_unique<uint32_t>(EquihashType::NHashWords);

        const uint32_t maxCollisionsPerBucket = 10000;

        // Allocate device buffer for collision pair pointers
        collisionPairs = make_cuda_unique<uint32_t*>(EquihashType::NBucketCount);
        vBucketCollisionPairs.resize(EquihashType::NBucketCount);
        for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
        {
            vBucketCollisionPairs[i] = make_cuda_unique<uint32_t>(maxCollisionsPerBucket);
            collisionPairs.get()[i] = vBucketCollisionPairs[i].get();
        }

        collisionCounters = make_cuda_unique<uint32_t*>(EquihashType::WK);

        vCollisionCounters.resize(EquihashType::WK);
        for (uint32_t round = 0; round < EquihashType::WK; ++round)
        {
            vCollisionCounters[round].resize(EquihashType::NBucketCount);
            for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
                vCollisionCounters[round][i] = make_cuda_unique<uint32_t>(1);
        }
        // Accumulated collision pair offsets for each round
        vCollisionPairsOffsets.resize(EquihashType::WK, 0);

        // Allocate device memory for solutions and solution count
        solutions = make_cuda_unique<typename EquihashType::solution>(MAXSOLUTIONS);
        solutionCount = make_cuda_unique<uint32_t>(1);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return false;
    }
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

    if (hashIdx >= EquihashType::NHashes)
        return;

    const uint32_t blockIndex = hashIdx / EquihashType::IndicesPerHashOutput;

    blake2b_state localState = *state;
    blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&blockIndex), sizeof(blockIndex));

    uint8_t hash[EquihashType::HashOutput];  
    blake2b_final_device(&localState, hash, EquihashType::HashOutput);

    const uint32_t outputIdx = hashIdx * EquihashType::HashWords;
    for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
        hashes[outputIdx + i] = (reinterpret_cast<uint32_t*>(hash))[i];
}

template<typename EquihashType>
void EhDevice<EquihashType>::generateInitialHashes()
{
    const uint32_t numThreads = (EquihashType::NHashes + ThreadsPerBlock - 1) / ThreadsPerBlock * ThreadsPerBlock;

    dim3 gridDim((numThreads + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 blockDim(ThreadsPerBlock);

    cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(initialState.get(), hashes.get());
}

template <typename EquihashType>
__global__ void cudaKernel_detectCollisions(
    const uint32_t* hashes, uint32_t* collisionPairs, uint32_t* collisionCounts,
    const uint32_t startIdx, const uint32_t endIdx, const uint32_t collisionBitMask)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx = startIdx + tid;

    if (idx >= endIdx)
        return;

    const uint32_t hash = hashes[idx];
    const uint32_t maskedHash = hash & collisionBitMask;

    const uint32_t bucketIdx = idx / EquihashType::NBucketSize;
    const uint32_t bucketOffset = bucketIdx * EquihashType::NBucketSize;

    for (uint32_t i = bucketOffset; i < idx; ++i)
    {
        const uint32_t otherHash = hashes[i];
        const uint32_t otherMaskedHash = otherHash & collisionBitMask;

        if (maskedHash == otherMaskedHash)
        {
            const uint32_t collisionIdx = atomicAdd(collisionCounts, 1);
            collisionPairs[collisionIdx] = (idx << 16) | i;
        }
    }    
}

template <typename EquihashType>
void EhDevice<EquihashType>::detectCollisions()
{
    // Calculate the accumulated collision pair offset for the current round
    uint32_t collisionPairsOffset = 0;
    if (round > 0)
        collisionPairsOffset = vCollisionPairsOffsets[round - 1];

    const uint32_t collisionBitLength = EquihashType::CollisionBitLength;
    const uint32_t collisionBitMask = ((1U << collisionBitLength) - 1) << (round * collisionBitLength);

    // Set collision counter pointers to the corresponding device buffers for the current round
    for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
        collisionCounters.get()[round * EquihashType::NBucketCount + i] = vCollisionCounters[round][i].get();

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        const uint32_t startIdx = bucketIdx * EquihashType::NBucketSize;
        const uint32_t endIdx = min(startIdx + EquihashType::NBucketSize, EquihashType::NHashes);
        const uint32_t numItems = endIdx - startIdx;

        const dim3 gridDim((numItems + ThreadsPerBlock - 1) / ThreadsPerBlock);
        const dim3 blockDim(ThreadsPerBlock);

        cudaKernel_detectCollisions<EquihashType><<<gridDim, blockDim>>>(
            hashes.get(), collisionPairs.get()[bucketIdx] + collisionPairsOffset,
            collisionCounters.get()[round * EquihashType::NBucketCount + bucketIdx], startIdx, endIdx, collisionBitMask);

        // Copy the collision count from device to host
        uint32_t collisionCount;
        copyToHost(&collisionCount, vCollisionCounters[round][bucketIdx].get(), sizeof(uint32_t));

        // Update the accumulated collision pair offset for the next bucket
        collisionPairsOffset += collisionCount;
    }

    // Store the accumulated collision pair offset for the current round
    vCollisionPairsOffsets[round] = collisionPairsOffset;
}

template<typename EquihashType>
__global__ void cudaKernel_xorCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    const uint32_t* const* collisionPairs, const uint32_t* const* collisionCounts,
    const uint32_t totalCollisionPairs)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= totalCollisionPairs)
        return;

    uint32_t collisionPairIdx = tid;
    uint32_t bucketIdx = 0;

    while (collisionPairIdx >= collisionCounts[bucketIdx][0] && bucketIdx < EquihashType::NBucketCount)
    {
        collisionPairIdx -= collisionCounts[bucketIdx][0];
        ++bucketIdx;
    }

    const uint32_t collisionPair = collisionPairs[bucketIdx][collisionPairIdx];
    const uint32_t idx1 = collisionPair >> 16;
    const uint32_t idx2 = collisionPair & 0xFFFF;

    for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
        xoredHashes[idx1 * EquihashType::HashWords + i] ^= hashes[idx2 * EquihashType::HashWords + i];
}

template<typename EquihashType>
void EhDevice<EquihashType>::xorCollisions()
{
    const uint32_t numBlocks = (vCollisionPairsOffsets[round] + ThreadsPerBlock - 1) / ThreadsPerBlock;

    const dim3 gridDim(numBlocks);
    const dim3 blockDim(ThreadsPerBlock);

    cudaKernel_xorCollisions<EquihashType><<<gridDim, blockDim>>>(
        hashes.get(), xoredHashes.get(), collisionPairs.get(), 
        collisionCounters.get() + round * EquihashType::NBucketCount,
        vCollisionPairsOffsets[round]);

    // Swap the hash pointers for the next round
    swap(hashes, xoredHashes);
}

/**
 * @brief Find valid solutions by checking the XORed values against the target difficulty.
 * 
 * @param hashes - Array of hash values 
 * @param solutions - Array to store the valid solutions
 * @param solutionCount - The number of valid solutions found 
 */
template<typename EquihashType>
__global__ void cudaKernel_findSolutions(
    const uint32_t* hashes,
    const uint32_t* const* collisionPairs,
    const uint32_t* const* collisionCounts,
    const uint32_t* collisionPairsOffsets, typename EquihashType::solution* solutions, uint32_t* solutionCount)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t totalCollisionPairs = collisionPairsOffsets[EquihashType::WK - 1];

    if (tid >= totalCollisionPairs)
        return;

    uint32_t collisionPairIdx = tid;
    uint32_t bucketIdx = 0;

    while (collisionPairIdx >= collisionCounts[EquihashType::WK - 1][bucketIdx] && bucketIdx < EquihashType::NBucketCount)
    {
        collisionPairIdx -= collisionCounts[EquihashType::WK - 1][bucketIdx];
        ++bucketIdx;
    }

    const uint32_t collisionPair = collisionPairs[bucketIdx][collisionPairsOffsets[EquihashType::WK - 2] + collisionPairIdx];
    uint32_t indices[EquihashType::ProofSize] = { 0 };
    uint32_t xoredHash[EquihashType::HashWords] = { 0 };

    indices[0] = collisionPair >> 16;
    indices[1] = collisionPair & 0xFFFF;

    for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
        xoredHash[i] = hashes[indices[0] * EquihashType::HashWords + i] ^ hashes[indices[1] * EquihashType::HashWords + i];

    uint32_t indicesCount = 2;

    for (uint32_t round = EquihashType::WK - 2; round >= 1; --round)
    {
        const uint32_t collisionPairsOffset = collisionPairsOffsets[round - 1];
        const uint32_t collisionPairsCount = collisionPairsOffsets[round] - collisionPairsOffset;

        bool found = false;

        for (uint32_t i = 0; i < collisionPairsCount; ++i)
        {
            const uint32_t pair = collisionPairs[bucketIdx][collisionPairsOffset + i];
            const uint32_t idx1 = pair >> 16;
            const uint32_t idx2 = pair & 0xFFFF;

            if (idx1 == indices[indicesCount - 2] || idx1 == indices[indicesCount - 1] ||
                idx2 == indices[indicesCount - 2] || idx2 == indices[indicesCount - 1])
            {
                const uint32_t newIndex = (idx1 == indices[indicesCount - 2] || idx1 == indices[indicesCount - 1]) ? idx2 : idx1;
                indices[indicesCount++] = newIndex;

                for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                    xoredHash[j] ^= hashes[newIndex * EquihashType::HashWords + j];

                found = true;
                break;
            }
        }

        if (!found)
            break;
    }

    if (indicesCount == EquihashType::ProofSize)
    {
        // Check if the xoredHash satisfies the difficulty target
        // TODO: Implement the difficulty check based on the specific target

        // If a valid solution is found, store it
        if (true/* Difficulty check passed */)
        {
            const uint32_t solutionIdx = atomicAdd(solutionCount, 1);
            for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
                solutions[solutionIdx].indices[i] = indices[i];
        }
    }
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::findSolutions()
{
    const uint32_t numBlocks = (vCollisionPairsOffsets[round] + ThreadsPerBlock - 1) / ThreadsPerBlock;

    cudaMemset(solutionCount.get(), 0, sizeof(uint32_t));

    const dim3 gridDim(numBlocks);
    const dim3 blockDim(ThreadsPerBlock);

    auto collisionPairsOffsets = make_cuda_unique<uint32_t>(vCollisionPairsOffsets.size());
    copyToDevice(collisionPairsOffsets.get(), vCollisionPairsOffsets.data(), vCollisionPairsOffsets.size() * sizeof(uint32_t));

    cudaKernel_findSolutions<EquihashType><<<gridDim, blockDim>>>(
        hashes.get(), collisionPairs.get(), collisionCounters.get(), collisionPairsOffsets.get(), solutions.get(), 
        solutionCount.get());

    uint32_t nSolutionCount;
    copyToHost(&nSolutionCount, solutionCount.get(), sizeof(uint32_t));

    return nSolutionCount;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintHashes()
{
    v_uint32 hostHashes(EquihashType::NHashWords);
    copyToHost(hostHashes.data(), hashes.get(), hostHashes.size() * sizeof(uint32_t));

    // Print out the generated hashes
    size_t hashNo = 0;
    for (size_t i = 0; i < hostHashes.size(); i += EquihashType::HashWords)
    {
        ++hashNo;
        if (hashNo % 0x1000 != 0)
            continue;
        cout << "Hash " << dec << hashNo << ": ";
        bool bAllZeroes = true;
        for (size_t j = 0; j < EquihashType::HashWords; ++j)
        {
            if (hostHashes[i + j])
                bAllZeroes = false;
            cout << hex << hostHashes[i + j] << " ";
        }
        if (bAllZeroes)
        {
            cout << "All zeroes" << endl;
            break;
        }
        cout << endl;
    }
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::solver()
{
    // Generate initial hash values
    generateInitialHashes();

    // Perform K rounds of collision detection and XORing
    for (uint32_t round = 0; round < EquihashType::WK; round++)
    {
        // Detect collisions and XOR the colliding pairs
        detectCollisions();
        xorCollisions();
    }

    return findSolutions();
}

template<typename EquihashType>
void EhDevice<EquihashType>::copySolutionsToHost(vector<typename EquihashType::solution> &vHostSolutions)
{
    uint32_t nSolutionCount = 0;
    copyToHost(&nSolutionCount, solutionCount.get(), sizeof(uint32_t));

    vHostSolutions.clear();
    // Resize the host solutions vector
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), solutions.get(), nSolutionCount * EquihashType::ProofSize);
}

// Explicit template instantiation
template class EhDevice<Eh200_9>;
