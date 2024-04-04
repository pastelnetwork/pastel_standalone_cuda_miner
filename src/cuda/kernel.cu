
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <vector>
#include <string>
#include <bitset>
#include <iostream>
#include <limits>
#include <iomanip>
#include <numeric>
#include <chrono>

#include <tinyformat.h>

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
        hashes = make_cuda_unique<uint32_t>(EquihashType::NHashStorageWords);
        // Allocate device memory for XORed hash values
        xoredHashes = make_cuda_unique<uint32_t>(EquihashType::NHashStorageWords);
        // Allocate device buffer for bucket hash indices
        bucketHashIndices = make_cuda_unique<uint32_t>(EquihashType::NHashStorageCount *(EquihashType::WK + 1));

        // Allocate device buffer for collision pair pointers
        collisionPairs = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * MaxCollisionsPerBucket);

        collisionCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));
        vCollisionCounters.resize(EquihashType::NBucketCount, 0);

        // Accumulated collision pair offsets for each bucket
        collisionPairOffsets = make_cuda_unique<uint32_t>(EquihashType::NBucketCount);
        vCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);
        vPrevCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);

        // Allocate device memory for solutions and solution count
        solutions = make_cuda_unique<typename EquihashType::solution_type>(MaxSolutions);
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
__global__ void cudaKernel_generateInitialHashes(const blake2b_state* state, uint32_t* hashes, const uint32_t numHashCalls)
{
    uint32_t hashIdx = (threadIdx.x + blockIdx.x * blockDim.x) * (numHashCalls * EquihashType::IndicesPerHashOutput);

    uint32_t i = 0;
    while ((hashIdx < EquihashType::NHashes) && (i++ < numHashCalls))
    {
        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashIdx), sizeof(hashIdx));

        uint8_t hash[EquihashType::HashOutput];  
        blake2b_final_device(&localState, hash, EquihashType::HashOutput);

        auto ph = reinterpret_cast<uint8_t*>(hash);
        uint32_t outputIdx = hashIdx * EquihashType::HashWords;
        constexpr uint32_t nSingleHashBytes = EquihashType::HashOutput / EquihashType::IndicesPerHashOutput;
        constexpr uint32_t nHashOutFullWords = nSingleHashBytes / sizeof(uint32_t);
        for (uint32_t j = 0; j < EquihashType::IndicesPerHashOutput; ++j)
        {
            for (uint32_t k = 0; k < nHashOutFullWords; ++k)
            {
                hashes[outputIdx + k] = 
                    (static_cast<uint32_t>(ph[3]) << 24) | 
                    (static_cast<uint32_t>(ph[2]) << 16) | 
                    (static_cast<uint32_t>(ph[1]) << 8) | 
                    static_cast<uint32_t>(ph[0]);
                ph += sizeof(uint32_t);
            }
            if (nHashOutFullWords < EquihashType::HashWords)
            {
                uint32_t nBytes = nSingleHashBytes - nHashOutFullWords * sizeof(uint32_t);
                uint32_t nWord = 0;
                for (uint32_t k = 0; k < nBytes; ++k)
                    nWord |= static_cast<uint32_t>(*ph++) << (k * 8);
                hashes[outputIdx + nHashOutFullWords] = nWord;
            }

            ++hashIdx;
            outputIdx += EquihashType::HashWords;
        }
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::generateInitialHashes()
{
    const uint32_t numHashCalls = EquihashType::NHashes / EquihashType::IndicesPerHashOutput;
    const uint32_t numHashCallsPerThread = 64;
    const uint32_t numThreads = (numHashCalls + numHashCallsPerThread - 1) / numHashCallsPerThread;

    dim3 gridDim((numThreads + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 blockDim(ThreadsPerBlock);

    cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(initialState.get(), hashes.get(), numHashCallsPerThread);
    cudaDeviceSynchronize();
}

template<typename EquihashType>
__global__ void cudaKernel_rebucketHashes(const uint32_t* hashes, uint32_t* bucketHashes,
    uint32_t* bucketHashIndices, uint32_t* collisionCounters,
    const bool bIsLastRound,
    const uint32_t bitOffset, const uint32_t wordOffset, const uint64_t rebucketMask,
    const uint32_t bucketOffset, const uint32_t bucketEndIdx)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx = bucketOffset + tid;
    if (idx >= bucketEndIdx)
        return;

    const uint32_t hashIdx = idx * EquihashType::HashWords;
    uint64_t combinedHash = (static_cast<uint64_t>(
        hashes[hashIdx + wordOffset + 1]) << 32) | 
        hashes[hashIdx + wordOffset];
    const uint64_t maskedHash = combinedHash & rebucketMask;
    uint32_t bucketIdx = maskedHash >> bitOffset;
    bucketIdx = bucketIdx < EquihashType::OverflowBucketIndex ? bucketIdx : EquihashType::OverflowBucketIndex;

    uint32_t newBucketOffset = bucketIdx * EquihashType::NBucketSize;
    uint32_t hashCounterPerBucket = atomicAdd(&collisionCounters[bucketIdx], 1);

    if (hashCounterPerBucket >= EquihashType::NBucketSize && bucketIdx != EquihashType::OverflowBucketIndex)
    {
        atomicSub(&collisionCounters[bucketIdx], 1);  // Decrement the collision counter of the original bucket

        bucketIdx = EquihashType::OverflowBucketIndex;
        newBucketOffset = bucketIdx * EquihashType::NBucketSize;
        hashCounterPerBucket = atomicAdd(&collisionCounters[bucketIdx], 1);
    }

    const uint32_t newBucketHashIndex = newBucketOffset + hashCounterPerBucket;
    const uint32_t bucketHashIdx = newBucketHashIndex * EquihashType::HashWords;
    if (bIsLastRound)
    {
        for (uint32_t i = 0; i < EquihashType::HashWords - 1; ++i)
            bucketHashes[bucketHashIdx + i] = 0;
        bucketHashes[bucketHashIdx + EquihashType::HashWords - 1] = combinedHash >> bitOffset;
    }
    else 
    {
        for (uint32_t i = 0; i < EquihashType::HashWords; ++i)
            bucketHashes[bucketHashIdx + i] = hashes[hashIdx + i];
    }

    bucketHashIndices[newBucketHashIndex] = idx;
}

template<typename EquihashType>
void EhDevice<EquihashType>::rebucketHashes()
{
    if (round == 0)
    {
        // Clear the first row of collisionCounters
        cudaMemset(collisionCounters.get(), 0, EquihashType::NBucketCount * sizeof(uint32_t));
        for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
            vCollisionCounters[i] = EquihashType::NBucketSize;
        vCollisionCounters[EquihashType::OverflowBucketIndex] = 
            EquihashType::NHashes - EquihashType::OverflowBucketIndex * EquihashType::NBucketSize;
    }

    const uint32_t globalBitOffset = round * EquihashType::CollisionBitLength;
    uint32_t wordOffset = globalBitOffset / numeric_limits<uint32_t>::digits;
    if (wordOffset >= EquihashType::HashWords - 1)
        wordOffset = EquihashType::HashWords - 2;
    const uint32_t bitOffset = globalBitOffset - wordOffset * numeric_limits<uint32_t>::digits;
    const uint64_t rebucketMask = ((1ULL << 5) - 1) << bitOffset;

    auto collisionCountersPtr = collisionCounters.get() + round * EquihashType::NBucketCount;
    cudaMemset(collisionCountersPtr, 0, EquihashType::NBucketCount * sizeof(uint32_t));
    auto bucketHashIndicesPtr = bucketHashIndices.get() + round * EquihashType::NHashStorageCount;

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; bucketIdx++)
    {
        const uint32_t bucketOffset = bucketIdx * EquihashType::NBucketSize;
        const uint32_t bucketEndIdx = bucketOffset + vCollisionCounters[bucketIdx];
        const uint32_t numItems = bucketEndIdx - bucketOffset;

        if (numItems == 0)
            continue;

        const dim3 gridDim((numItems + ThreadsPerBlock - 1) / ThreadsPerBlock);
        const dim3 blockDim(ThreadsPerBlock);
    
        cudaKernel_rebucketHashes<EquihashType><<<gridDim, blockDim>>>(
            hashes.get(), xoredHashes.get(), bucketHashIndicesPtr, collisionCountersPtr,
            (round == EquihashType::WK), bitOffset, wordOffset, rebucketMask, bucketOffset, bucketEndIdx);

            DEBUG_FN(debugPrintBucketCounters(bucketIdx, collisionCountersPtr));
    }
    cudaDeviceSynchronize();

    copyToHost(vCollisionCounters.data(), collisionCountersPtr, EquihashType::NBucketCount * sizeof(uint32_t));
}

template <typename EquihashType>
void EhDevice<EquihashType>::debugPrintBucketCounters(const uint32_t bucketIdx, const uint32_t *collisionCountersPtr)
{
    v_uint32 vBucketCollisionCounts(EquihashType::NBucketCount);
    copyToHost(vBucketCollisionCounts.data(), collisionCountersPtr, vBucketCollisionCounts.size() * sizeof(uint32_t));

    cout << "Round #" << dec << round << " [buckets " << dec << (bucketIdx + 1) << "/" << EquihashType::NBucketCount << "] hashes: ";
    for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
        cout << vBucketCollisionCounts[i] << " ";
    cout << endl;
}

__device__ bool atomicCheckAndIncrement(uint32_t* address, const uint32_t limit, uint32_t *oldValue)
{
    uint32_t old = *address;
    uint32_t assumed;

    do {
        assumed = old;
        if (assumed >= limit)
        {
            if (oldValue)
                *oldValue = assumed; // Write the old value back through the pointer

            return false; // Indicate that we did not increment because it's at/above the limit
        }

        old = atomicCAS(address, assumed, assumed + 1);
    } while (assumed != old);

    if (oldValue)
        *oldValue = assumed; // Successfully incremented, write the old value back
    return true; // Indicate success
}

template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* collisionPairs, uint32_t* collisionCounts,
    const uint32_t bucketOffset, const uint32_t bucketEndIdx, 
    const uint32_t prevTotalCollisionPairs,
    const uint32_t wordOffset, const uint64_t collisionBitMask)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t idx = bucketOffset + tid;

    if (idx >= bucketEndIdx)
        return;

    const uint32_t hashBaseIdx = idx * EquihashType::HashWords;
    const uint32_t hashWordIdx = hashBaseIdx + wordOffset;
    const uint64_t maskedHash = 
        ((static_cast<uint64_t>(hashes[hashWordIdx + 1]) << 32) | 
                                hashes[hashWordIdx]) & collisionBitMask;

    uint32_t leftPairIdx = (idx - bucketOffset) & 0xFFFF;
    for (uint32_t i = bucketOffset; i < idx; ++i)
    {
        const uint32_t otherHashBaseIdx = i * EquihashType::HashWords;
        const uint32_t otherHashWordIdx = otherHashBaseIdx + wordOffset;
        const uint64_t otherMaskedHash = 
            ((static_cast<uint64_t>(hashes[otherHashWordIdx + 1]) << 32) | 
                                    hashes[otherHashWordIdx]) & collisionBitMask;


        if (maskedHash == otherMaskedHash)
        {
            // hash collision found - xor the hashes and store the result
            uint32_t rightPairIdx = (i - bucketOffset) & 0xFFFF;
            if (leftPairIdx == rightPairIdx)
                continue;
            uint32_t xoredHash[EquihashType::HashWords];
            bool bAllZeroes = true;
            for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
            {
                xoredHash[j] = hashes[hashBaseIdx + j] ^ hashes[otherHashBaseIdx + j];
                if (xoredHash[j])
                    bAllZeroes = false;
            }
            if (bAllZeroes)
                continue;

            uint32_t collisionIdx = 0;
            if (atomicCheckAndIncrement(collisionCounts, EquihashType::NBucketSize, &collisionIdx))
            {
                collisionPairs[collisionIdx] = (leftPairIdx << 16) | rightPairIdx;

                const uint32_t xoredHashBaseIdx = (bucketOffset + collisionIdx) * EquihashType::HashWords;
                for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                    xoredHashes[xoredHashBaseIdx + j] = xoredHash[j];
            }
        }
    }    
}

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
    const uint32_t globalBitOffset = round * EquihashType::CollisionBitLength;
    uint32_t wordOffset = globalBitOffset / numeric_limits<uint32_t>::digits;
    if (wordOffset >= EquihashType::HashWords - 1)
        wordOffset = EquihashType::HashWords - 2;
    const uint32_t bitOffset = globalBitOffset - wordOffset * numeric_limits<uint32_t>::digits;
    const uint64_t collisionBitMask = ((1ULL << EquihashType::CollisionBitLength) - 1) << bitOffset;

    auto collisionCountersBasePtr = collisionCounters.get() + round * EquihashType::NBucketCount;
    cudaMemset(collisionCountersBasePtr, 0, EquihashType::NBucketCount * sizeof(uint32_t));

    vPrevCollisionPairsOffsets = vCollisionPairsOffsets;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; bucketIdx++)
    {
        const uint32_t bucketOffset = bucketIdx * EquihashType::NBucketSize;
        const uint32_t bucketEndIdx = bucketOffset + vCollisionCounters[bucketIdx];
        const uint32_t numItems = bucketEndIdx - bucketOffset;
        
        if (numItems == 0)
            continue;

        const dim3 gridDim((numItems + ThreadsPerBlock - 1) / ThreadsPerBlock);
        const dim3 blockDim(ThreadsPerBlock);

        try
        {
            auto collisionPairsPtr = collisionPairs.get() + bucketIdx * MaxCollisionsPerBucket + vCollisionPairsOffsets[bucketIdx];
            auto collisionCountersPtr = collisionCountersBasePtr + bucketIdx;
            cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                hashes.get(), xoredHashes.get(),
                collisionPairsPtr, collisionCountersPtr,
                bucketOffset, bucketEndIdx, 
                vCollisionPairsOffsets[bucketIdx], 
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

            vCollisionCounters[bucketIdx] = collisionCount;
            // Store the accumulated collision pair offset for the current round
            vCollisionPairsOffsets[bucketIdx] += collisionCount;

            DEBUG_FN(debugPrintCollisionCounter(bucketOffset, bucketEndIdx, bucketIdx, collisionCount));
        } 
        catch (const exception& e)
        {
            cerr << "Exception in processCollisions: " << e.what() << endl;
        }
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintCollisionCounter(const uint32_t bucketOffset, const uint32_t bucketEndIdx,
    const uint32_t bucketIdx, const uint32_t collisionCount)
{
    cout << "Round #" << dec << round << 
    " [" << bucketOffset << " - " << bucketEndIdx << "]" <<
    " Bucket #" << dec << bucketIdx << 
    " Collisions: " << dec << collisionCount << 
    " Total collisions: " << dec << vCollisionPairsOffsets[bucketIdx] << endl;
}

template<typename EquihashType>
__global__ void cudaKernel_findSolutions(
    const uint32_t* hashes,
    const uint32_t* collisionPairs,
    const uint32_t* collisionPairOffsets,
    const uint32_t* collisionCounters,
    const uint32_t* bucketHashIndices,
    typename EquihashType::solution_type* solutions, uint32_t* solutionCount,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t maxSolutionCount,
    const uint32_t bucketIdx, 
    const uint32_t bucketOffset,
    const uint32_t bucketSize)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t gid = bucketOffset + tid;

    if (tid >= bucketSize)
        return;

    uint32_t curCollisionPairOffsets[EquihashType::NBucketCount];

    uint32_t indices[EquihashType::ProofSize];
    uint32_t indicesNew[EquihashType::ProofSize];
    uint32_t *pIndices = indices;
    uint32_t *pIndicesNew = indicesNew;

    const uint32_t lastHashWord = hashes[gid * EquihashType::HashWords + EquihashType::HashWords - 1];
    for (uint32_t idx = bucketOffset; idx < gid; ++idx)
    {
        const uint32_t lastHashWordOther = hashes[idx * EquihashType::HashWords + EquihashType::HashWords - 1];
        if (lastHashWordOther != lastHashWord)
            continue;
        
        // found solution
        printf("[%u] Found solution [%u-%u] \n", tid, idx, gid);
        auto bucketHashIndicesPtr = bucketHashIndices + EquihashType::WK * EquihashType::NHashStorageCount + 
            bucketIdx * EquihashType::NBucketSize;
        pIndices[0] = bucketHashIndicesPtr[tid];
        uint32_t numIndices = 1;
        uint32_t numIndicesNew = 0;

        const auto curCollisionCountersPtr = collisionCounters + EquihashType::WK * EquihashType::NBucketCount;
        for (uint32_t n = 0; n < EquihashType::NBucketCount; ++n)
            curCollisionPairOffsets[n] = collisionPairOffsets[n] - curCollisionCountersPtr[n];

        for (int round = EquihashType::WK - 1; round >= 0; --round)
        {
            const auto bucketHashIndicesRoundPtr = bucketHashIndices + round * EquihashType::NHashStorageCount;
            const auto curCollisionCountersRoundPtr = collisionCounters + round * EquihashType::NBucketCount;

            for (uint32_t index = 0; index < numIndices; ++index)
            {
                const uint32_t curIndex = pIndices[index];
                const uint32_t curBucketIdx = curIndex / EquihashType::NBucketSize;
                const uint32_t curIdx = curIndex - curBucketIdx * EquihashType::NBucketSize;
                const auto collisionPairsPtr = collisionPairs + curBucketIdx * maxCollisionsPerBucket + curCollisionPairOffsets[curBucketIdx];
                const uint32_t curCollisionPair = collisionPairsPtr[curIdx];
                const uint32_t index1 = curCollisionPair >> 16;
                const uint32_t index2 = curCollisionPair & 0xFFFF;
                bucketHashIndicesPtr = bucketHashIndicesRoundPtr + curBucketIdx * EquihashType::NBucketSize;
                pIndicesNew[numIndicesNew++] = bucketHashIndicesPtr[index1];
                pIndicesNew[numIndicesNew++] = bucketHashIndicesPtr[index2];
            }
            uint32_t *pIndicesTemp = pIndices;
            pIndices = pIndicesNew;
            pIndicesNew = pIndicesTemp;
            numIndices = numIndicesNew;
            numIndicesNew = 0;
            for (uint32_t n = 0; n < EquihashType::NBucketCount; ++n)
                curCollisionPairOffsets[n] -= curCollisionCountersRoundPtr[n];
        }

        uint32_t solutionIndex = 0;
        if (atomicCheckAndIncrement(solutionCount, maxSolutionCount, &solutionIndex))
        {
            for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
                solutions[solutionIndex].indices[i] = indices[i];
        }    
    }
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::findSolutions()
{
    uint32_t numSolutions = 0;
    cudaMemset(solutionCount.get(), 0, sizeof(uint32_t));

    copyToDevice(collisionPairOffsets.get(), vCollisionPairsOffsets.data(), EquihashType::NBucketCount * sizeof(uint32_t));

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        const uint32_t bucketOffset = bucketIdx * EquihashType::NBucketSize;
        const uint32_t bucketSize = vCollisionCounters[bucketIdx];

        if (bucketSize == 0)
            continue;

        const dim3 gridDim((bucketSize + ThreadsPerBlock - 1) / ThreadsPerBlock);
        const dim3 blockDim(ThreadsPerBlock);

        cudaKernel_findSolutions<EquihashType><<<gridDim, blockDim>>>(
            hashes.get(), 
            collisionPairs.get(),
            collisionPairOffsets.get(),
            collisionCounters.get(),
            bucketHashIndices.get(),
            solutions.get(), solutionCount.get(),
            MaxCollisionsPerBucket,
            MaxSolutions,
            bucketIdx, bucketOffset, bucketSize);
    }
    cudaDeviceSynchronize();

    copyToHost(&numSolutions, solutionCount.get(), sizeof(uint32_t));

    return numSolutions;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintHashes(const bool bIsBucketed)
{
    v_uint32 hostHashes;
    const uint32_t nHashStorageWords = bIsBucketed ? EquihashType::NHashStorageWords : EquihashType::NHashWords;
    hostHashes.resize(nHashStorageWords);
    copyToHost(hostHashes.data(), hashes.get(), hostHashes.size() * sizeof(uint32_t));

    size_t nHashCount = bIsBucketed ? accumulate(vCollisionCounters.cbegin(), vCollisionCounters.cend(), 0) : EquihashType::NHashes;
    cout << (bIsBucketed ? "Bucketed hashes" : "Initial hashes") << 
        " (" << dec << nHashCount << "):" << endl;
    // Print out the hashes
    size_t hashTotalNo = 0;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t bucketSize = bIsBucketed ? vCollisionCounters[bucketIdx] : EquihashType::NBucketSize;
        size_t bucketOffset = bucketIdx * EquihashType::NBucketSize * EquihashType::HashWords;

        // correct size for the last bucket
        if (!bIsBucketed && (bucketIdx == EquihashType::NBucketCount - 1))
            bucketSize = EquihashType::NHashes - bucketIdx * EquihashType::NBucketSize;
        cout << "Bucket " << dec << bucketIdx << " (" << bucketSize << " hashes):" << endl;

        size_t hashNo = 0;
        for (size_t i = 0; i < bucketSize; ++i)
        {
            ++hashNo;
            ++hashTotalNo;
            if (hashNo > 5 && (hashNo % 0x1000 != 0))
                continue;

            size_t hashOffset = bucketOffset + i * EquihashType::HashWords;
            cout << "Hash " << dec << hashTotalNo << ": " << hex << setfill('0');
            bool bAllZeroes = true;
            for (size_t j = 0; j < EquihashType::HashWords; ++j)
            {
                if (hostHashes[hashOffset + j])
                    bAllZeroes = false;
                cout << setw(8) << hostHashes[hashOffset + j] << " ";
            }
            if (bAllZeroes)
            {
                cout << "All zeroes !!!" << endl;
                break;
            }
            cout << endl;
        }
        cout << dec;
    }
    // print hashes count in each bucket
    if (bIsBucketed)
    {
        cout << "Bucket sizes: " << dec;
        for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
            cout << vCollisionCounters[i] << " ";
        cout << endl;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintXoredHashes()
{
    v_uint32 hostHashes;
    cout << endl << "Xored hashes for round #" << round << ":" << endl;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t nBucketCollisionCount = vCollisionCounters[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
        cout << "Bucket #" << dec << bucketIdx << " (" << 
            nBucketCollisionCount << ") xored hashes: " << endl;

        hostHashes.resize(nBucketCollisionCount * EquihashType::HashWords);
        copyToHost(hostHashes.data(), xoredHashes.get() + bucketIdx * EquihashType::NBucketSize * EquihashType::HashWords,
            nBucketCollisionCount * EquihashType::HashWords * sizeof(uint32_t));

        size_t hashNo = 0;
        for (uint32_t i = 0; i < nBucketCollisionCount; ++i)
        {
            ++hashNo;
            if (hashNo > 10 && (hashNo % 100 != 0))
                 continue;
            cout << "Hash " << dec << hashNo << ": " << hex << setfill('0') ;
            bool bAllZeroes = true;
            for (size_t j = 0; j < EquihashType::HashWords; ++j)
            {
                const uint32_t hashIdx = i * EquihashType::HashWords + j;
                if (hostHashes[hashIdx])
                    bAllZeroes = false;
                cout << setw(8) << hostHashes[hashIdx] << " ";
            }
            if (bAllZeroes)
            {
                cout << "All zeroes !!!" << endl;
//                break;
            }
            cout << endl;
            if (hashNo % 64 == 0)
            {
                cout << endl;
                continue;
            }
        }
        cout << dec;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintCollisionPairs()
{
    v_uint32 vBucketCollisionCounts(EquihashType::NBucketCount);
    copyToHost(vBucketCollisionCounts.data(),
        collisionCounters.get() + EquihashType::NBucketCount * round,
        vBucketCollisionCounts.size() * sizeof(uint32_t));

    constexpr uint32_t COLLISIONS_PER_LINE = 10;
    cout << endl << "Collision pairs for round #" << round << ":" << endl;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t nBucketCollisionCount = vBucketCollisionCounts[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
        cout << "Bucket #" << dec << bucketIdx << " (" << nBucketCollisionCount << ") collision pairs: " << endl;

        v_uint32 hostCollisionPairs(nBucketCollisionCount);
        copyToHost(hostCollisionPairs.data(), collisionPairs.get() + bucketIdx * MaxCollisionsPerBucket + vPrevCollisionPairsOffsets[bucketIdx], nBucketCollisionCount * sizeof(uint32_t));

        uint32_t nPairNo = 0;
        for (uint32_t i = 0; i < nBucketCollisionCount; ++i)
        {
            ++nPairNo;
            if (nPairNo > 30)
                break;
            uint32_t collisionPair = hostCollisionPairs[i];
            uint32_t idx1 = collisionPair >> 16;
            uint32_t idx2 = collisionPair & 0xFFFF;
            if (i % COLLISIONS_PER_LINE == 0)
            {
                if (i > 0)
                    cout << endl;
                cout << "Pair " << dec << i << ":";
            }
            cout << " (" << idx1 << "," << idx2 << ")";
        }
        cout << endl;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::copySolutionsToHost(vector<typename EquihashType::solution_type> &vHostSolutions)
{
    uint32_t nSolutionCount = 0;
    copyToHost(&nSolutionCount, solutionCount.get(), sizeof(uint32_t));

    // Resize the host solutions vector
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), solutions.get(), nSolutionCount * EquihashType::ProofSize);
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::solver()
{
    // Generate initial hash values
    EQUI_TIMER_DEFINE;
    EQUI_TIMER_START;
    generateInitialHashes();
    EQUI_TIMER_STOP("Initial hash generation");
    DEBUG_FN(debugPrintHashes(false));

    // Perform K rounds of collision detection and XORing
    while (round < EquihashType::WK)
    {
        EQUI_TIMER_START;
        rebucketHashes();
        EQUI_TIMER_STOP(strprintf("Round [%u], rebucketing", round));

        swap(hashes, xoredHashes);
        DEBUG_FN(debugPrintHashes(true));

        // Detect collisions and XOR the colliding hashes
        EQUI_TIMER_START;
        processCollisions();
        EQUI_TIMER_STOP(strprintf("Round [%u], collisions", round));
        DEBUG_FN(debugPrintCollisionPairs());
        DEBUG_FN(debugPrintXoredHashes());

        // Swap the hash pointers for the next round
        swap(hashes, xoredHashes);
        ++round;
        cout << "Round #" << dec << round << " completed" << endl;
    }
    EQUI_TIMER_START;
    rebucketHashes();
    EQUI_TIMER_STOP(strprintf("Round [%u], rebucketing", round));
    swap(hashes, xoredHashes);
    DEBUG_FN(debugPrintHashes(true));

    return findSolutions();
}

// Explicit template instantiation
template class EhDevice<EquihashSolver<200, 9>>;
