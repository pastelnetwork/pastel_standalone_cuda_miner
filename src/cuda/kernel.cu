
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <vector>
#include <string>
#include <bitset>
#include <iostream>
#include <limits>

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

        // Allocate device buffer for collision pair pointers
        collisionPairs = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * MaxCollisionsPerBucket);

        collisionCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));

        // Accumulated collision pair offsets for each bucket
        vCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);
        vPrevCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);
        collisionPairOffsets = make_cuda_unique<uint32_t>(EquihashType::NBucketCount);

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
    cudaDeviceSynchronize();
}

template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* collisionPairs, uint32_t* collisionCounts,
    const uint32_t startIdx, const uint32_t endIdx, const uint32_t prevTotalCollisionPairs,
    const uint32_t wordOffset, const uint64_t collisionBitMask)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx = startIdx + tid;

    if (idx >= endIdx)
        return;

    const uint32_t hashBaseIdx = idx * EquihashType::HashWords;
    const uint32_t hashWordIdx = hashBaseIdx + wordOffset;
    const uint64_t maskedHash = 
        ((static_cast<uint64_t>(hashes[hashWordIdx]) << 32) | 
                                hashes[hashWordIdx + 1]) & collisionBitMask;

    const uint32_t bucketIdx = idx / EquihashType::NBucketSize;
    const uint32_t bucketOffset = bucketIdx * EquihashType::NBucketSize;

    for (uint32_t i = bucketOffset; i < idx; ++i)
    {
        const uint32_t otherHashBaseIdx = i * EquihashType::HashWords;
        const uint32_t otherHashWordIdx = otherHashBaseIdx + wordOffset;
        const uint64_t otherMaskedHash = 
            ((static_cast<uint64_t>(hashes[otherHashWordIdx]) << 32) | 
                                    hashes[otherHashWordIdx + 1]) & collisionBitMask;

        if (maskedHash == otherMaskedHash)
        {
            const uint32_t collisionIdx = prevTotalCollisionPairs + atomicAdd(collisionCounts, 1);
            collisionPairs[collisionIdx] = (idx << 16) | i;

            const uint32_t xoredHashBaseIdx = bucketOffset + collisionIdx * EquihashType::HashWords;
            // hash collision found - xor the hashes and store the result
            for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                xoredHashes[xoredHashBaseIdx + j] = hashes[hashBaseIdx + j] ^ hashes[otherHashBaseIdx + j];
        }
    }    
}

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
    const uint32_t collisionBitLength = EquihashType::CollisionBitLength;

    const uint32_t globalBitOffset = round * collisionBitLength;
    uint32_t wordOffset = globalBitOffset / numeric_limits<uint32_t>::digits;
    if (wordOffset >= EquihashType::HashWords - 1)
        wordOffset = EquihashType::HashWords - 2;
    const uint32_t bitOffset = numeric_limits<uint32_t>::digits - collisionBitLength - (globalBitOffset % numeric_limits<uint32_t>::digits);
    const uint64_t collisionBitMask = ((1ULL << collisionBitLength) - 1) << bitOffset;

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        const uint32_t startIdx = bucketIdx * EquihashType::NBucketSize;
        const uint32_t endIdx = min(startIdx + EquihashType::NBucketSize, EquihashType::NHashes);
        const uint32_t numItems = endIdx - startIdx;

        const dim3 gridDim((numItems + ThreadsPerBlock - 1) / ThreadsPerBlock, EquihashType::NBucketCount);
        const dim3 blockDim(ThreadsPerBlock);

        try
        {
            auto collisionPairsPtr = collisionPairs.get() + bucketIdx * MaxCollisionsPerBucket + vCollisionPairsOffsets[bucketIdx];
            auto collisionCountersPtr = collisionCounters.get() + bucketIdx * EquihashType::WK + round;
            cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                hashes.get(), xoredHashes.get(),
                collisionPairsPtr, collisionCountersPtr,
                startIdx, endIdx, vPrevCollisionPairsOffsets[bucketIdx],
                wordOffset, collisionBitMask);

            // Check for any CUDA errors
            cudaError_t cudaError = cudaGetLastError();
            if (cudaError != cudaSuccess)
            {
                cerr << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
                throw runtime_error("CUDA kernel launch failed");
            }
            cudaDeviceSynchronize();
                
            // Copy the collision count from device to host
            uint32_t collisionCount = 0;
            copyToHost(&collisionCount, collisionCountersPtr, sizeof(uint32_t));

            // Store the accumulated collision pair offset for the current round
            vCollisionPairsOffsets[bucketIdx] += collisionCount;

            cout << "Round #" << dec << round << 
                    " Bucket #" << dec << bucketIdx << 
                    " Collisions: " << dec << collisionCount << 
                    " Total collisions: " << dec << vCollisionPairsOffsets[bucketIdx] << endl;
        } 
        catch (const exception& e)
        {
            cerr << "Exception in detectCollisions: " << e.what() << endl;
        }
    }
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
    const uint32_t* collisionPairs,
    const uint32_t* collisionCounts,
    const uint32_t* collisionPairsOffsets,
    typename EquihashType::solution* solutions, uint32_t* solutionCount,
    const uint32_t maxCollisionsPerBucket)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t totalCollisionPairs = collisionPairsOffsets[EquihashType::WK - 1];

    if (tid >= totalCollisionPairs)
        return;

    uint32_t collisionPairIdx = tid;
    uint32_t bucketIdx = 0;

    while (collisionPairIdx >= collisionCounts[bucketIdx * EquihashType::WK + EquihashType::WK - 1] && bucketIdx < EquihashType::NBucketCount)
    {
        collisionPairIdx -= collisionCounts[bucketIdx * EquihashType::WK + EquihashType::WK - 1];
        ++bucketIdx;
    }

    const uint32_t collisionPair = collisionPairs[bucketIdx * maxCollisionsPerBucket + totalCollisionPairs + collisionPairIdx];
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
            const uint32_t pair = collisionPairs[bucketIdx * maxCollisionsPerBucket + collisionPairsOffset + i];
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
    const uint32_t totalCollisionPairs = vCollisionPairsOffsets[round];
    const uint32_t numBlocks = (totalCollisionPairs + ThreadsPerBlock - 1) / ThreadsPerBlock;

    cudaMemset(solutionCount.get(), 0, sizeof(uint32_t));

    const dim3 gridDim(numBlocks);
    const dim3 blockDim(ThreadsPerBlock);

    auto collisionPairsOffsets = make_cuda_unique<uint32_t>(vCollisionPairsOffsets.size());
    copyToDevice(collisionPairsOffsets.get(), vCollisionPairsOffsets.data(), vCollisionPairsOffsets.size() * sizeof(uint32_t));

    cudaKernel_findSolutions<EquihashType><<<gridDim, blockDim>>>(
        hashes.get(),
        collisionPairs.get(),
        collisionCounters.get(),
        collisionPairsOffsets.get(),
        solutions.get(), 
        solutionCount.get(),
        MaxCollisionsPerBucket);

    uint32_t nSolutionCount;
    copyToHost(&nSolutionCount, solutionCount.get(), sizeof(uint32_t));

    return nSolutionCount;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintHashes()
{
    v_uint32 hostHashes;
    hostHashes.resize(EquihashType::NHashWords);
    copyToHost(hostHashes.data(), hashes.get(), hostHashes.size() * sizeof(uint32_t));

    cout << "Initial hashes (" << hostHashes.size() << "):" << endl;
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
            cout << "All zeroes !!!" << endl;
            break;
        }
        cout << endl;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintXoredHashes()
{
    v_uint32 hostHashes;
    v_uint32 vBucketCollisionCounts;
    vBucketCollisionCounts.resize(EquihashType::NBucketCount * (EquihashType::WK + 1));
    copyToHost(vBucketCollisionCounts.data(), collisionCounters.get(), vBucketCollisionCounts.size() * sizeof(uint32_t));
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t hashNo = 0;
        // auto collisionCountersPtr = collisionCounters.get() + bucketIdx * EquihashType::WK + round;
        cout << "Bucket #" << dec << bucketIdx << ": " << endl;
        for (uint32_t i = 0; i < vBucketCollisionCounts[bucketIdx * EquihashType::WK + round]; ++i)
        {
            ++hashNo;
            if (hashNo % 0x1000 != 0)
                continue;
            cout << "Hash " << dec << hashNo << ": ";
            bool bAllZeroes = true;
            for (size_t j = 0; j < EquihashType::HashWords; ++j)
            {
                const uint32_t hashIdx = bucketIdx * EquihashType::NBucketSize + i * EquihashType::HashWords + j;
                if (hostHashes[hashIdx])
                    bAllZeroes = false;
                cout << hex << hostHashes[hashIdx] << " ";
            }
            if (bAllZeroes)
            {
                cout << "All zeroes !!!" << endl;
                break;
            }
            cout << endl;
        }
    }
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::solver()
{
    // Generate initial hash values
    generateInitialHashes();
    debugPrintHashes();

    // Perform K rounds of collision detection and XORing
    for (uint32_t round = 0; round < EquihashType::WK; round++)
    {
        vPrevCollisionPairsOffsets = vCollisionPairsOffsets;

        // Detect collisions and XOR the colliding pairs
        processCollisions();
        debugPrintXoredHashes();

        // Swap the hash pointers for the next round
        swap(hashes, xoredHashes);
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
