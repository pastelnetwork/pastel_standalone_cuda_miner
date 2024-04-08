
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

#include <src/kernel/kernel.h>
#include <src/kernel/memutils.h>
#include <src/kernel/blake2b_device.h>
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
        bucketHashCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));

        // Allocate device buffer for collision pair pointers
        collisionPairs = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * MaxCollisionsPerBucket);

        collisionCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));
        vCollisionCounters.resize(EquihashType::NBucketCount, 0);
        discardedCounter = make_cuda_unique<uint32_t>(1);

        // collision pair offsets for each bucket for each round
        collisionOffsets = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));
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

// CUDA kernel to generate initial hashes from blake2b state
template<typename EquihashType>
__global__ void cudaKernel_generateInitialHashes(const blake2b_state* state, uint32_t* hashes, 
    uint32_t *bucketHashIndices, uint32_t *bucketHashCounters, uint32_t *discardedCounter,
    const uint32_t numHashCalls)
{
    const uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t hashInputIdx = tid * numHashCalls;

    uint8_t hash[EquihashType::HashOutput];  
    uint32_t i = 0;
    while ((hashInputIdx < EquihashType::Base) && (i++ < numHashCalls))
    {
        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashInputIdx), sizeof(hashInputIdx));
        blake2b_final_device(&localState, hash, EquihashType::HashOutput);

        uint32_t curHashIdx = hashInputIdx * EquihashType::IndicesPerHashOutput;
        const uint32_t curBucketIdx = curHashIdx / EquihashType::NBucketSize;
        uint32_t hashOffset = 0;
        for (uint32_t j = 0; j < EquihashType::IndicesPerHashOutput; ++j)
        {
            // map the output hash index to the appropriate bucket
            // and store the hash in the corresponding bucket
            // index format: [F][CC][B BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
            // F - flip collision pair, C = collision local index, B = bucket index, N = hash index
            const uint16_t bucketIdx = (static_cast<uint16_t>(hash[hashOffset + 1]) << 8 | hash[hashOffset]) & EquihashType::NBucketIdxMask;
            
            uint32_t hashIdxInBucket = 0;
            if (!atomicCheckAndIncrement(&bucketHashCounters[bucketIdx], EquihashType::NBucketSize, &hashIdxInBucket))
            {
                atomicAdd(discardedCounter, 1);
                continue;
            }

            const uint32_t bucketHashIdx = bucketIdx * EquihashType::NBucketSize + hashIdxInBucket;
            const uint32_t bucketHashIdxPtr = bucketHashIdx * EquihashType::HashWords;

            const uint32_t curHashIdxInBucket = curHashIdx % EquihashType::NBucketSize;
            const uint32_t bucketHashStoredIdx = curBucketIdx << 16 | curHashIdxInBucket;
            bucketHashIndices[bucketHashIdx] = bucketHashStoredIdx;

            for (uint32_t k = 0; k < EquihashType::HashFullWords; ++k)
            {
                hashes[bucketHashIdxPtr + k] = 
                    (static_cast<uint32_t>(hash[hashOffset + 3]) << 24) | 
                    (static_cast<uint32_t>(hash[hashOffset + 2]) << 16) | 
                    (static_cast<uint32_t>(hash[hashOffset + 1]) << 8) | 
                    static_cast<uint32_t>(hash[hashOffset]);
                hashOffset += sizeof(uint32_t);
            }
            if (EquihashType::HashPartialBytesLeft > 0)
            {
                uint32_t nWord = 0;
                for (uint32_t k = 0; k < EquihashType::HashPartialBytesLeft; ++k)
                    nWord |= static_cast<uint32_t>(hash[hashOffset++]) << (k * 8);
                hashes[bucketHashIdxPtr + EquihashType::HashFullWords] = nWord;
            }
            ++curHashIdx;
        }
        ++hashInputIdx;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::generateInitialHashes()
{
    const uint32_t numHashCalls = EquihashType::NHashes / EquihashType::IndicesPerHashOutput;
    const uint32_t numHashCallsPerThread = EquihashType::NBucketSize;
    const uint32_t numThreads = (numHashCalls + numHashCallsPerThread - 1) / numHashCallsPerThread;
    
    dim3 gridDim((numThreads + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 blockDim(ThreadsPerBlock);
    
    cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(initialState.get(), hashes.get(), 
        bucketHashIndices.get(), bucketHashCounters.get(), discardedCounter.get(), numHashCallsPerThread);
    cudaDeviceSynchronize();
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


template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* bucketHashIndices,
    const uint32_t* bucketHashCountersPrev,
    uint32_t* bucketHashCounters,
    uint32_t* collisionPairs, 
    const uint32_t* collisionOffsets,
    uint32_t* collisionCounters,
    uint32_t* discardedCounter,
    const uint32_t bitOffset, const uint32_t wordOffset, const uint64_t collisionBitMask)
{
    const uint32_t bucketIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const uint32_t startIdx = bucketIdx * EquihashType::NBucketSize;
    const uint32_t endIdx = startIdx + bucketHashCountersPrev[bucketIdx];
    uint32_t xoredHash[EquihashType::HashWords];

    uint32_t hashIdxLeft = (startIdx + 1) * EquihashType::HashWords;
    for (uint32_t idxLeft = startIdx + 1; idxLeft < endIdx; ++idxLeft)
    {
        // each collision info holds up to 3 collision pairs
        // first bytes points to the left pair, the next 3 bytes point to the right collision pairs
        const uint32_t leftPairIdx = (idxLeft - startIdx) & 0xFF;
        uint32_t collisionInfo = leftPairIdx;
        uint32_t collisionInfoIdx = 0;
        const uint32_t hashWordIdxLeft = hashIdxLeft + wordOffset;
        const uint64_t hashLeft = 
            ((static_cast<uint64_t>(hashes[hashWordIdxLeft + 1]) << 32) | 
                                    hashes[hashWordIdxLeft]);
        const uint64_t maskedHashLeft = hashLeft & collisionBitMask;
        
        uint32_t hashIdxRight = startIdx * EquihashType::HashWords;
        for (uint32_t idxRight = startIdx;  idxRight < idxLeft; ++idxRight)
        {
            const uint32_t hashWordIdxRight = hashIdxRight + wordOffset;
            const uint64_t hashRight = (static_cast<uint64_t>(hashes[hashWordIdxRight + 1]) << 32) | 
                                                             hashes[hashWordIdxRight];
            const uint64_t maskedHashRight = hashRight & collisionBitMask;
            if (maskedHashLeft == maskedHashRight)
            {
                uint32_t rightPairIdx = (idxRight - startIdx) & 0xFF;
                // hash collision found - xor the hashes and store the result
                if (leftPairIdx == rightPairIdx)
                    continue;
                bool bAllZeroes = true;
                for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                {
                    xoredHash[j] = hashes[hashIdxLeft + j] ^ hashes[hashIdxRight + j];
                    if (xoredHash[j])
                        bAllZeroes = false;
                }
                if (bAllZeroes)
                    continue; // skip if all zeroes
                // define xored hash bucket based on the first NBucketIdxMask bits (starting from the bitOffset)
                const uint32_t xoredBucketIdx = 
                    (static_cast<uint32_t>(((static_cast<uint64_t>(xoredHash[1]) << 32) | xoredHash[0]) >> EquihashType::CollisionBitLength))
                    & EquihashType::NBucketIdxMask;
                uint32_t xoredHashIdxInBucket = 0;
                if (!atomicCheckAndIncrement(&bucketHashCounters[xoredBucketIdx], EquihashType::NBucketSize, &xoredHashIdxInBucket))
                {
                    atomicAdd(discardedCounter, 1);
                    continue; // skip if the bucket is full
                }
                // index format: [F][CC][B BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // F - flip collision pair, C = collision local index, B = bucket index, N = collision pair index
                const uint32_t xoredBucketHashIdx = xoredBucketIdx * EquihashType::NBucketSize + xoredHashIdxInBucket;
                const uint32_t xoredBucketHashIdxPtr = xoredBucketHashIdx * EquihashType::HashWords;
                for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                    xoredHashes[xoredBucketHashIdxPtr + j] = xoredHash[j];
                uint32_t collisionIndex = ((hashLeft < hashRight ? 0 : 1) << 2) | (collisionInfoIdx + 1);
                collisionIndex <<= EquihashType::NBucketIdxBits;
                collisionIndex |= bucketIdx;
                collisionIndex <<= sizeof(uint16_t);
                collisionIndex |= collisionCounters[bucketIdx];
                bucketHashIndices[xoredBucketHashIdx] = collisionIndex;

                collisionInfo <<= 8;
                collisionInfo |= rightPairIdx;
                ++collisionInfoIdx;

                if (collisionInfoIdx >= 3)
                {
                    const uint32_t collisionPairIdx = collisionOffsets[bucketIdx] + collisionCounters[bucketIdx];
                    collisionPairs[collisionPairIdx] = collisionInfo;
                    collisionCounters[bucketIdx] += 1;
                    collisionInfo = leftPairIdx;
                    collisionInfoIdx = 0;
                }
            }

            hashIdxRight += EquihashType::HashWords;
        }
        if (collisionInfoIdx > 0)
        {
            collisionInfo <<= 8 * (3 - collisionInfoIdx);
            const uint32_t collisionPairIdx = collisionOffsets[bucketIdx] + collisionCounters[bucketIdx];
            collisionPairs[collisionPairIdx] = collisionInfo;
            collisionCounters[bucketIdx] += 1;
        }

        hashIdxLeft += EquihashType::HashWords;
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

//    dim3 gridDim((EquihashType::NBucketCount + ThreadsPerBlock - 1) / ThreadsPerBlock);
//    dim3 blockDim(ThreadsPerBlock);

    dim3 gridDim(10);
    dim3 blockDim(1);

    try {
        uint32_t* collisionCountersPtr = collisionCounters.get() + round * EquihashType::NBucketCount;

        cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                    hashes.get(), xoredHashes.get(),
                    bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
                    bucketHashCounters.get() + round * EquihashType::NBucketCount,
                    bucketHashCounters.get() + (round + 1) * EquihashType::NBucketCount,
                    collisionPairs.get(),
                    collisionOffsets.get() + round * EquihashType::NBucketCount, 
                    collisionCountersPtr,
                    discardedCounter.get(),
                    bitOffset, wordOffset, collisionBitMask);

        // Check for any CUDA errors
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            cerr << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
            throw runtime_error("CUDA kernel launch failed");
        }
        cudaDeviceSynchronize();

        // Copy the collision counters from device to host
        copyToHost(vCollisionCounters.data(), collisionCountersPtr, EquihashType::NBucketCount * sizeof(uint32_t));

        // Store the accumulated collision pair offset for the current round
        for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
            vCollisionPairsOffsets[bucketIdx] += vCollisionCounters[bucketIdx];
        
        copyToDevice(collisionOffsets.get() + round * EquihashType::NBucketCount, 
            vCollisionPairsOffsets.data(), EquihashType::NBucketCount * sizeof(uint32_t));

    } catch (const exception& e)
    {
        cerr << "Exception in processCollisions: " << e.what() << endl;
    }
}

template<typename EquihashType>
__global__ void cudaKernel_findSolutions(
    const uint32_t* hashes,
    const uint32_t* bucketHashCounters,
    const uint32_t* bucketHashIndices,
    const uint32_t* collisionPairs,
    const uint32_t* collisionOffsets,
    const uint32_t* collisionCounters,
    typename EquihashType::solution_type* solutions, uint32_t* solutionCount,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t maxSolutionCount)
{
    const uint32_t bucketIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const auto bucketHashCountersLast = bucketHashCounters + EquihashType::NBucketCount * EquihashType::WK;
    const uint32_t startIdx = bucketIdx * EquihashType::NBucketSize;
    const uint32_t endIdx = startIdx + bucketHashCountersLast[bucketIdx];

    uint32_t indices[EquihashType::ProofSize];
    uint32_t indicesNew[EquihashType::ProofSize];
    uint32_t *pIndices = indices;
    uint32_t *pIndicesNew = indicesNew;

    const auto curCollisionCountersPtr = collisionCounters + EquihashType::WK * EquihashType::NBucketCount;
    const auto curCollisionOffsetsPtr = collisionOffsets + EquihashType::WK * EquihashType::NBucketCount;
 
    for (uint32_t idxLeft = startIdx + 1; idxLeft < endIdx; ++idxLeft)
    {
        const uint32_t hashIdxLeft = idxLeft * EquihashType::HashWords;        
        const uint32_t lastHashWordLeft = hashes[hashIdxLeft + EquihashType::HashWords - 1];

        for (uint32_t idxRight = startIdx; idxRight < idxLeft; ++idxRight)
        {
            const uint32_t lastHashWordRight = hashes[idxRight * EquihashType::HashWords + EquihashType::HashWords - 1];
            if (lastHashWordRight != lastHashWordLeft)
                continue;
            
            // found solution
            printf("[%u] Found solution [%u-%u] \n", bucketIdx, idxLeft, idxRight);
            auto bucketHashIndicesBasePtr = bucketHashIndices + EquihashType::WK * EquihashType::NHashStorageCount;

            indices[0] = idxLeft;
            indices[1] = idxRight;
            uint32_t numIndices = 2;
            uint32_t numIndicesNew = 0;

            for (int round = EquihashType::WK - 1; round >= 0; --round)
            {
                const auto bucketHashIndicesRoundPtr = bucketHashIndices + round * EquihashType::NHashStorageCount;
                const auto curCollisionCountersRoundPtr = collisionCounters + round * EquihashType::NBucketCount;
                const auto curCollisionOffsetsRoundPtr = collisionOffsets + round * EquihashType::NBucketCount;

                for (uint32_t index = 0; index < numIndices; ++index)
                {
                    const uint32_t curIndex = pIndices[index];
                    const uint32_t curBucketIdx = curIndex / EquihashType::NBucketSize;
                    const uint32_t curIdx = curIndex % EquihashType::NBucketSize;
                    const auto bucketHashIndicesPtr = bucketHashIndicesRoundPtr + curBucketIdx * EquihashType::NBucketSize;
                    // pointer to the collision pair format: [F][CC][B BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                    // F - flip collision pair, C = collision local index, B = bucket index, N = collision pair index
                    const auto ptr = bucketHashIndicesPtr[curIdx];
                    const auto collisionPairIndex = ptr & 0xFFFF;
                    const auto ptrHighWord = ptr >> 16;
                    const auto collisionPairBucketIdx = ptrHighWord & EquihashType::NBucketIdxMask;
                    const auto collisionPairHighBits = ptrHighWord >> EquihashType::NBucketIdxBits;
                    const auto collisionPairNum = collisionPairHighBits & 0x3;
                    const auto collisionPairFlip = collisionPairHighBits >> 2;

                    const auto collisionPairsPtr = collisionPairs + 
                        collisionPairBucketIdx * maxCollisionsPerBucket + 
                        curCollisionOffsetsRoundPtr[collisionPairBucketIdx];
                    const uint32_t collisionPairInfo = collisionPairsPtr[collisionPairIndex];
                    // collision pair info:
                    // 3 bytes: right collision pair indices
                    // 4th byte: left collision pair index
                    uint32_t leftPairIdx = (collisionPairInfo >> 24) & 0xFF;
                    uint32_t rightPairIdx = (collisionPairInfo >> (3 - collisionPairNum) * 8) & 0xFF;
                    if (collisionPairFlip == 1)
                    {
                        auto temp = leftPairIdx;
                        leftPairIdx = rightPairIdx;
                        rightPairIdx = temp;
                    }

                    pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + leftPairIdx;
                    pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + rightPairIdx;
                }
                uint32_t *pIndicesTemp = pIndices;
                pIndices = pIndicesNew;
                pIndicesNew = pIndicesTemp;
                numIndices = numIndicesNew;
                numIndicesNew = 0;
            }

            uint32_t solutionIndex = 0;
            if (atomicCheckAndIncrement(solutionCount, maxSolutionCount, &solutionIndex))
            {
                for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
                    solutions[solutionIndex].indices[i] = indices[i];
            }    
        }
    }
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::findSolutions()
{
    uint32_t numSolutions = 0;
    cudaMemset(solutionCount.get(), 0, sizeof(uint32_t));

    const dim3 gridDim((EquihashType::NBucketCount + ThreadsPerBlock - 1) / ThreadsPerBlock);
    const dim3 blockDim(ThreadsPerBlock);

    cudaKernel_findSolutions<EquihashType><<<gridDim, blockDim>>>(
        hashes.get(),
        bucketHashCounters.get(),
        bucketHashIndices.get(),
        collisionPairs.get(),
        collisionOffsets.get(),
        collisionCounters.get(),
        solutions.get(), solutionCount.get(),
        MaxCollisionsPerBucket,
        MaxSolutions);
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

        if (bucketIdx > 3 && bucketIdx < EquihashType::NBucketCount - 3)
            continue;
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
    uint32_t nDiscarded = 0;
    copyToHost(&nDiscarded, discardedCounter.get(), sizeof(uint32_t));
    cout << "Discarded: " << dec << nDiscarded << endl;
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
                const uint32_t hashInputIdx = i * EquihashType::HashWords + j;
                if (hostHashes[hashInputIdx])
                    bAllZeroes = false;
                cout << setw(8) << hostHashes[hashInputIdx] << " ";
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

    return findSolutions();
}

// Explicit template instantiation
template class EhDevice<EquihashSolver<200, 9>>;
