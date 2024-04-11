
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
        collisionCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount);
        vCollisionCounters.resize(EquihashType::NBucketCount, 0);

        discardedCounter = make_cuda_unique<uint32_t>(1);

        // collision pair offsets for each bucket for each round
        collisionOffsets = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));
        vCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);

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
            // index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
            //   B = bucket index, N = hash index
            const uint16_t bucketIdx = (static_cast<uint16_t>(hash[hashOffset + 1]) << 8 | hash[hashOffset]) & EquihashType::NBucketIdxMask;
            
            uint32_t hashIdxInBucket = 0;
            if (!atomicCheckAndIncrement(&bucketHashCounters[bucketIdx], EquihashType::NBucketSizeExtra, &hashIdxInBucket))
            {
                atomicAdd(discardedCounter, 1);
                continue;
            }
            // find the place where to store the hash (extra space exists in each bucket)
            const uint32_t bucketStorageHashIdx = bucketIdx * EquihashType::NBucketSizeExtra + hashIdxInBucket;
            const uint32_t bucketStorageHashIdxPtr = bucketStorageHashIdx * EquihashType::HashWords;

            // hash index to store uint32_t (bucket index [16] | hash index in the bucket [16])
            const uint32_t curHashIdxInBucket = curHashIdx % EquihashType::NBucketSize;
            const uint32_t bucketHashStoredIdx = (curBucketIdx << 16) | curHashIdxInBucket;
            bucketHashIndices[bucketStorageHashIdx] = bucketHashStoredIdx;

            for (uint32_t k = 0; k < EquihashType::HashFullWords; ++k)
            {
                hashes[bucketStorageHashIdxPtr + k] = 
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
                hashes[bucketStorageHashIdxPtr + EquihashType::HashFullWords] = nWord;
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
void EhDevice<EquihashType>::debugWriteBucketIndices()
{
    if (!m_dbgFile.is_open())
    {
        m_dbgFile.open("equihash_dbg.txt", ios::out);
        if (!m_dbgFile.is_open())
            return;
    }

    v_uint32 vBucketHashCounters(EquihashType::NBucketCount, 0);
    copyToHost(vBucketHashCounters.data(), bucketHashCounters.get() + round * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount);
    copyToHost(vBucketHashIndices.data(), bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
        EquihashType::NHashStorageCount * sizeof(uint32_t));

    m_dbgFile << "------------------------------------------------------------\n";
    m_dbgFile << endl << "Bucket hash indices for round #" << round << ":" << endl;
    for (size_t i = 0; i < EquihashType::NBucketCount; ++i)
    {
        if (vBucketHashCounters[i] == 0)
            continue;
        m_dbgFile << strprintf("\nRound %u, Bucket #%u, %u hash indices: ", round, i, vBucketHashCounters[i]);
        for (size_t j = 0; j < vBucketHashCounters[i]; ++j)
        {
            const uint32_t bucketHashIdx = vBucketHashIndices[i * EquihashType::NBucketSizeExtra + j];
            const uint32_t bucketIdx = bucketHashIdx >> 16;
            const uint32_t hashIdxInBucket = bucketHashIdx & 0xFFFF;
            if (j % 20 == 0)
                m_dbgFile << endl << "#" << dec << j << ": ";
            m_dbgFile << strprintf("(%u-%u) ", bucketIdx, hashIdxInBucket);
        }
        m_dbgFile << endl;
    }
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

__device__ void uint32ToString(uint32_t value, char* buffer, int* offset)
{
    if (value == 0)
    {
        buffer[(*offset)++] = '0';
        return;
    }

    char temp[10];
    int len = 0;
    while (value != 0)
    {
        temp[len++] = '0' + (value % 10);
        value /= 10;
    }

    for (int i = len - 1; i >= 0; i--)
    {
        buffer[(*offset)++] = temp[i];
    }
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
    uint32_t* discardedCounter, const uint32_t maxCollisionsPerBucket,
    const uint32_t wordOffset, const uint64_t collisionBitMask,
    const uint32_t xoredBitOffset, const uint32_t xoredWordOffset)
{
    const uint32_t bucketIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const auto collisionPairsPtr = collisionPairs + bucketIdx * maxCollisionsPerBucket;
    const auto collisionBucketOffset = collisionOffsets[bucketIdx];
    const uint32_t startIdxStorage = bucketIdx * EquihashType::NBucketSizeExtra;
    const uint32_t endIdxStorage = startIdxStorage + bucketHashCountersPrev[bucketIdx];
    uint32_t xoredHash[EquihashType::HashWords];

    uint32_t hashIdxLeft = (startIdxStorage + 1) * EquihashType::HashWords;
    for (uint32_t idxLeft = startIdxStorage + 1; idxLeft < endIdxStorage; ++idxLeft)
    {
        // each collision info holds up to 2 collision pairs
        // first one points to the left pair, the next 2 point to the right collision pairs
        const uint32_t leftPairIdx = idxLeft - startIdxStorage;
        const uint32_t hashWordIdxLeft = hashIdxLeft + wordOffset;
        const uint64_t hashLeft = 
            ((static_cast<uint64_t>(hashes[hashWordIdxLeft + 1]) << 32) | 
                                    hashes[hashWordIdxLeft]);
        const uint64_t maskedHashLeft = hashLeft & collisionBitMask;
        
        for (uint32_t idxRight = startIdxStorage;  idxRight < idxLeft; ++idxRight)
        {
            const uint32_t hashIdxRight = idxRight * EquihashType::HashWords;
            const uint32_t hashWordIdxRight = hashIdxRight + wordOffset;
            const uint64_t hashRight = (static_cast<uint64_t>(hashes[hashWordIdxRight + 1]) << 32) | 
                                                             hashes[hashWordIdxRight];
            const uint64_t maskedHashRight = hashRight & collisionBitMask;
            if (maskedHashLeft == maskedHashRight)
            {
                const uint32_t rightPairIdx = idxRight - startIdxStorage;
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
                // define xored hash bucket based on the first NBucketIdxMask bits (starting from the CollisionBitLength)                
                const uint32_t xoredBucketIdx = 
                    (static_cast<uint32_t>(((static_cast<uint64_t>(xoredHash[xoredWordOffset + 1]) << 32) | xoredHash[xoredWordOffset]) >> xoredBitOffset))
                    & EquihashType::NBucketIdxMask;
                uint32_t xoredHashIdxInBucket = 0;
                if (!atomicCheckAndIncrement(&bucketHashCounters[xoredBucketIdx], EquihashType::NBucketSizeExtra, &xoredHashIdxInBucket))
                {
                    atomicAdd(discardedCounter, 1);
                    continue; // skip if the bucket is full
                }
                // index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // B = bucket index, N = collision pair index
                const uint32_t xoredBucketHashIdxStorage = xoredBucketIdx * EquihashType::NBucketSizeExtra + xoredHashIdxInBucket;
                const uint32_t xoredBucketHashIdxStoragePtr = xoredBucketHashIdxStorage * EquihashType::HashWords;
                for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                    xoredHashes[xoredBucketHashIdxStoragePtr + j] = xoredHash[j];
                const uint32_t collisionPairIdx = collisionBucketOffset + collisionCounters[bucketIdx];
                bucketHashIndices[xoredBucketHashIdxStorage] = bucketIdx << 16 | collisionPairIdx;
                if (hashLeft < hashRight)
                    collisionPairsPtr[collisionPairIdx] = (rightPairIdx << 16) | leftPairIdx;
                else
                    collisionPairsPtr[collisionPairIdx] = (leftPairIdx << 16) | rightPairIdx;
                collisionCounters[bucketIdx] += 1;
            }
        }

        hashIdxLeft += EquihashType::HashWords;
    }
}

// template <typename EquihashType>
// void EhDevice<EquihashType>::debugWriteCollisionCounters()
// {
    // debug print the collision counters
    // char debugBuffer[6 * EquihashType::NBucketCount];
    // int offset = 0;
    // for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
    // {
    //     uint32ToString(collisionCounters[i], debugBuffer, &offset);
    //     debugBuffer[offset++] = ' ';
    // }
    // printf("Round #%u [bucket %u] collision counters: %s\n", blockIdx.x, bucketIdx, debugBuffer);
// }

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
    const uint32_t globalBitOffset = round * EquihashType::CollisionBitLength;
    uint32_t wordOffset = globalBitOffset / numeric_limits<uint32_t>::digits;
    if (wordOffset >= EquihashType::HashWords - 1)
        wordOffset = EquihashType::HashWords - 2;
    const uint32_t bitOffset = globalBitOffset - wordOffset * numeric_limits<uint32_t>::digits;
    const uint64_t collisionBitMask = ((1ULL << EquihashType::CollisionBitLength) - 1) << bitOffset;

    const uint32_t xoredGlobalBitOffset = globalBitOffset + EquihashType::CollisionBitLength;
    uint32_t xoredWordOffset = xoredGlobalBitOffset / numeric_limits<uint32_t>::digits;
    if (xoredWordOffset >= EquihashType::HashWords - 1)
        xoredWordOffset = EquihashType::HashWords - 2;
    const uint32_t xoredBitOffset = xoredGlobalBitOffset - xoredWordOffset * numeric_limits<uint32_t>::digits;

    cudaMemset(discardedCounter.get(), 0, sizeof(uint32_t));

    dim3 gridDim((EquihashType::NBucketCount + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 blockDim(ThreadsPerBlock);

    try {
        cudaMemset(collisionCounters.get(), 0, EquihashType::NBucketCount * sizeof(uint32_t));

        cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                    hashes.get(), xoredHashes.get(),
                    bucketHashIndices.get() + (round + 1) * EquihashType::NHashStorageCount,
                    bucketHashCounters.get() + round * EquihashType::NBucketCount,
                    bucketHashCounters.get() + (round + 1) * EquihashType::NBucketCount,
                    collisionPairs.get(),
                    collisionOffsets.get() + round * EquihashType::NBucketCount, 
                    collisionCounters.get(),
                    discardedCounter.get(), MaxCollisionsPerBucket,
                    wordOffset, collisionBitMask,
                    xoredBitOffset, xoredWordOffset);

        // Check for any CUDA errors
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            cerr << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
            throw runtime_error("CUDA kernel launch failed");
        }
        cudaDeviceSynchronize();

        // Copy the collision counters from device to host
        copyToHost(vCollisionCounters.data(), collisionCounters.get(), EquihashType::NBucketCount * sizeof(uint32_t));

        // Store the accumulated collision pair offset for the current round
        for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
            vCollisionPairsOffsets[bucketIdx] += vCollisionCounters[bucketIdx];
        
        copyToDevice(collisionOffsets.get() + (round + 1) * EquihashType::NBucketCount, vCollisionPairsOffsets.data(),
            EquihashType::NBucketCount * sizeof(uint32_t));

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
    typename EquihashType::solution_type* solutions, uint32_t* solutionCount,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t maxSolutionCount)
{
    const uint32_t bucketIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const auto bucketHashCountersLast = bucketHashCounters + EquihashType::WK * EquihashType::NBucketCount;
    const uint32_t startIdxStorage = bucketIdx * EquihashType::NBucketSizeExtra;
    const uint32_t endIdxStorage = startIdxStorage + bucketHashCountersLast[bucketIdx];

    uint32_t indices[EquihashType::ProofSize];
    uint32_t indicesNew[EquihashType::ProofSize];
    uint32_t *pIndices = indices;
    uint32_t *pIndicesNew = indicesNew;

    for (uint32_t idxStorage = startIdxStorage; idxStorage < endIdxStorage; ++idxStorage)
    {
        const uint32_t hashIdx = idxStorage * EquihashType::HashWords;        
        const uint32_t lastHashWord = hashes[hashIdx + EquihashType::HashWords - 1];

        if (lastHashWord != 0)
            continue;

        // found solution
        const uint32_t idxInBucket = idxStorage - startIdxStorage;
        printf("Found solution [%u-%u] \n", bucketIdx, idxInBucket);

        indices[0] = bucketIdx * EquihashType::NBucketSize + idxInBucket;
        uint32_t numIndices = 1;

        for (int round = EquihashType::WK - 1; round >= 0; --round)
        {
            const auto bucketHashIndicesRoundPtr = bucketHashIndices + (round + 1) * EquihashType::NHashStorageCount;

            uint32_t numIndicesNew = 0;
            for (uint32_t index = 0; index < numIndices; ++index)
            {
                // pointer to the collision pair format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // B = bucket index, N = collision pair index
                const auto idx = pIndices[index];
                const auto idxBucket = idx >> 16;
                const auto idxInBucket = idx & 0xFFFF;

                const auto storageIdx = idxBucket * EquihashType::NBucketSizeExtra + idxInBucket;
                const auto ptr = bucketHashIndicesRoundPtr[storageIdx];
                const auto collisionPairBucketIdx = ptr >> 16;
                const auto collisionPairIndex = ptr & 0xFFFF;

                const auto collisionPairsPtr = collisionPairs + collisionPairBucketIdx * maxCollisionsPerBucket;
                const uint32_t collisionPairInfo = collisionPairsPtr[collisionPairIndex];
                const uint32_t pairIdx1 = collisionPairInfo >> 16;
                const uint32_t pairIdx2 = collisionPairInfo & 0xFFFF;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx1;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx2;
            }
            uint32_t *pIndicesTemp = pIndices;
            pIndices = pIndicesNew;
            pIndicesNew = pIndicesTemp;
            numIndices = numIndicesNew;
        }
        
        uint32_t solutionIndex = 0;
        if (atomicCheckAndIncrement(solutionCount, maxSolutionCount, &solutionIndex))
        {
            // map to the original indices
            for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
            {
                const auto idx = pIndices[i];
                const auto idxBucket = idx >> 16;
                const auto idxInBucket = idx & 0xFFFF;
                const auto storageIdx = idxBucket * EquihashType::NBucketSizeExtra + idxInBucket;
                const uint32_t index = bucketHashIndices[storageIdx];
                const uint32_t origBucketIdx = index >> 16;
                const uint32_t origIdx = index & 0xFFFF;
                solutions[solutionIndex].indices[i] = origBucketIdx * EquihashType::NBucketSize + origIdx;
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
        solutions.get(), solutionCount.get(),
        MaxCollisionsPerBucket,
        MaxSolutions);
    cudaDeviceSynchronize();

    copyToHost(&numSolutions, solutionCount.get(), sizeof(uint32_t));

    return numSolutions;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugWriteHashes(const bool bInitialHashes)
{
    if (!m_dbgFile.is_open())
    {
        m_dbgFile.open("equihash_dbg.txt", ios::out);
        if (!m_dbgFile.is_open())
            return;
    }

    m_dbgFile << "------------------------------------------------------------\n";
    v_uint32 vBucketHashCounters(EquihashType::NBucketCount, 0);
    copyToHost(vBucketHashCounters.data(), bucketHashCounters.get() + 
        (round + (bInitialHashes ? 0 : 1)) * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vHostHashes;
    vHostHashes.resize(EquihashType::NHashStorageWords);
    copyToHost(vHostHashes.data(), hashes.get(), EquihashType::NHashStorageWords * sizeof(uint32_t));

    uint32_t nDiscarded = 0;
    copyToHost(&nDiscarded, discardedCounter.get(), sizeof(uint32_t));

    v_uint32 hostHashes;
    string sLog;
    sLog.reserve(1024);
    m_dbgFile << strprintf("\nHashes for round #%u (discarded - %u):\n", round, nDiscarded);

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t nBucketHashCount = vBucketHashCounters[bucketIdx];
        if (nBucketHashCount == 0)
            continue;
        sLog = strprintf("\nRound %u Bucket #%u, (%u) hashes: \n", round, bucketIdx, nBucketHashCount);            
        m_dbgFile << sLog;

        const uint32_t bucketHashStorageIdx = bucketIdx * EquihashType::NBucketSizeExtra;
        size_t hashNo = 0;
        for (uint32_t i = 0; i < nBucketHashCount; ++i)
        {
            sLog = strprintf("Hash %u: ", hashNo);
            bool bAllZeroes = true;
            for (size_t j = 0; j < EquihashType::HashWords; ++j)
            {
                const uint32_t hashInputIdx = (bucketHashStorageIdx + i) * EquihashType::HashWords + j;
                if (vHostHashes[hashInputIdx])
                    bAllZeroes = false;
                sLog += strprintf("%08x ", vHostHashes[hashInputIdx]);
            }
            if (bAllZeroes)
                sLog += " (all zeroes)";
            sLog += "\n";
            m_dbgFile << sLog;
            ++hashNo;
        }
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintHashes(const bool bInitialHashes)
{
    v_uint32 vBucketHashCounters(EquihashType::NBucketCount, 0);
    copyToHost(vBucketHashCounters.data(), bucketHashCounters.get() + 
        (round + (bInitialHashes ? 0 : 1)) * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    uint32_t nDiscarded = 0;
    copyToHost(&nDiscarded, discardedCounter.get(), sizeof(uint32_t));
    cout << "Discarded: " << dec << nDiscarded << endl;

    v_uint32 vHostHashes;
    vHostHashes.resize(EquihashType::NHashStorageWords);
    copyToHost(vHostHashes.data(), hashes.get(), EquihashType::NHashStorageWords * sizeof(uint32_t));
    
    v_uint32 hostHashes;
    string sLog;
    sLog.reserve(1024);
    sLog = strprintf("\nHashes for round #%u (discarded - %u):\n", round, nDiscarded);
    cout << sLog;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        if ((bucketIdx > 3) && (bucketIdx < EquihashType::NBucketCount - 3))
            continue;
        size_t nBucketHashCount = vBucketHashCounters[bucketIdx];
        if (nBucketHashCount == 0)
            continue;
        cout << strprintf("\nRound %u Bucket #%u, (%u) hashes: \n", round, bucketIdx, nBucketHashCount);

        const uint32_t bucketStorageIdx = bucketIdx * EquihashType::NBucketSizeExtra;
        size_t hashNo = 0;
        for (uint32_t i = 0; i < nBucketHashCount; ++i)
        {
            if (hashNo > 5 && (hashNo < nBucketHashCount - 5))
            {
                ++hashNo;
                continue;
            }
            sLog = strprintf("Hash %u (%u): ", hashNo, bucketIdx * EquihashType::NBucketSize + i);
            bool bAllZeroes = true;
            for (size_t j = 0; j < EquihashType::HashWords; ++j)
            {
                const uint32_t hashInputIdx = (bucketStorageIdx + i) * EquihashType::HashWords + j;
                if (vHostHashes[hashInputIdx])
                    bAllZeroes = false;
                sLog += strprintf("%08x ", vHostHashes[hashInputIdx]);
            }
            if (bAllZeroes)
                sLog += " (all zeroes)";
            sLog += "\n";
            cout << sLog;
            ++hashNo;
        }
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugWriteCollisionPairs()
{
    v_uint32 vBucketCollisionCounts(EquihashType::NBucketCount, 0);
    v_uint32 vCollisionPairsOffsets(EquihashType::NBucketCount, 0);
    copyToHost(vBucketCollisionCounts.data(),
        collisionCounters.get(), EquihashType::NBucketCount * sizeof(uint32_t));

    copyToHost(vCollisionPairsOffsets.data(),
        collisionOffsets.get() + round * EquihashType::NBucketCount,
        vCollisionPairsOffsets.size() * sizeof(uint32_t));

    v_uint32 vCollisionPairs(EquihashType::NBucketCount * MaxCollisionsPerBucket);
    copyToHost(vCollisionPairs.data(), 
        collisionPairs.get(), vCollisionPairs.size() * sizeof(uint32_t));

    constexpr uint32_t COLLISIONS_PER_LINE = 10;
    m_dbgFile << "------------------------------------------------------------\n";
    m_dbgFile << endl << "Collision pairs for round #" << round << ":" << endl;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t nBucketCollisionCount = vBucketCollisionCounts[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
        m_dbgFile << strprintf("\nRound %u, Bucket #%u, collision pairs %u (bucket offsets: %u...%u), collision bucket origin: %u:\n",
            round, bucketIdx, nBucketCollisionCount, vCollisionPairsOffsets[bucketIdx], vCollisionPairsOffsets[bucketIdx] + nBucketCollisionCount,
            bucketIdx * MaxCollisionsPerBucket);

        uint32_t nPairNo = 0;
        for (uint32_t i = 0; i < nBucketCollisionCount; ++i)
        {
            ++nPairNo;
            if (nPairNo > 30)
                break;
            const uint32_t collisionPairInfo = vCollisionPairs[bucketIdx * MaxCollisionsPerBucket +
                vCollisionPairsOffsets[bucketIdx] + i];
            const uint32_t idxLeft = collisionPairInfo >> 16;
            const uint32_t idxRight = collisionPairInfo & 0xFFFF;
            if (i % COLLISIONS_PER_LINE == 0)
            {
                if (i > 0)
                    cout << endl;
                m_dbgFile << endl << "PairInfo " << dec << i << ":";
            }
            m_dbgFile << " (" << idxLeft << "-" << idxRight << ")";
        }
        m_dbgFile << endl;
    }
    // find max collision pair offset & count
    uint32_t maxCollisionPairOffset = 0;
    uint32_t maxCollisionPairCount = 0;
    for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
    {
        if (vCollisionPairsOffsets[i] > maxCollisionPairOffset)
            maxCollisionPairOffset = vCollisionPairsOffsets[i];
        if (vBucketCollisionCounts[i] > maxCollisionPairCount)
            maxCollisionPairCount = vBucketCollisionCounts[i];
    }
    m_dbgFile << "Max collision pair offset: " << dec << maxCollisionPairOffset << ", max collision pair count: " << maxCollisionPairCount << endl;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugPrintCollisionPairs()
{
    v_uint32 vBucketCollisionCounts(EquihashType::NBucketCount, 0);
    v_uint32 vCollisionPairsOffsets(EquihashType::NBucketCount, 0);
    copyToHost(vBucketCollisionCounts.data(),
        collisionCounters.get(),
        vBucketCollisionCounts.size() * sizeof(uint32_t));
    if (round > 0)
        copyToHost(vCollisionPairsOffsets.data(),
            collisionOffsets.get() + round * EquihashType::NBucketCount,
            vCollisionPairsOffsets.size() * sizeof(uint32_t));

    constexpr uint32_t COLLISIONS_PER_LINE = 10;
    cout << endl << "Collision pairs for round #" << round << ":" << endl;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        if ((bucketIdx > 3) && (bucketIdx < EquihashType::NBucketCount - 3))
            continue;
        size_t nBucketCollisionCount = vBucketCollisionCounts[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
        cout << strprintf("\nRound %u, Bucket #%u, %u collision pairs:\n", round, bucketIdx, nBucketCollisionCount);  

        v_uint32 hostCollisionPairs(nBucketCollisionCount);
        copyToHost(hostCollisionPairs.data(), 
            collisionPairs.get() + bucketIdx * MaxCollisionsPerBucket + vCollisionPairsOffsets[bucketIdx], 
            nBucketCollisionCount * sizeof(uint32_t));

        uint32_t nPairNo = 0;
        for (uint32_t i = 0; i < nBucketCollisionCount; ++i)
        {
            ++nPairNo;
            if (nPairNo > 30)
                break;
            const uint32_t collisionPairInfo = hostCollisionPairs[i];
            const uint32_t idxLeft = collisionPairInfo >> 16;
            const uint32_t idxRight = collisionPairInfo & 0xFFFF;
            if (i % COLLISIONS_PER_LINE == 0)
            {
                if (i > 0)
                    cout << endl;
                cout << "PairInfo " << dec << i << ":";
            }
            cout << " (" << idxLeft << "-" << idxRight << ")";
        }
        cout << endl;
    }
    // find max collision pair offset & count
    uint32_t maxCollisionPairOffset = 0;
    uint32_t maxCollisionPairCount = 0;
    for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
    {
        if (vCollisionPairsOffsets[i] > maxCollisionPairOffset)
            maxCollisionPairOffset = vCollisionPairsOffsets[i];
        if (vBucketCollisionCounts[i] > maxCollisionPairCount)
            maxCollisionPairCount = vBucketCollisionCounts[i];
    }
    cout << "Max collision pair offset: " << dec << maxCollisionPairOffset << ", max collision pair count: " << maxCollisionPairCount << endl;
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugWriteSolutions(const vector<typename EquihashType::solution_type>& vHostSolutions)
{
    m_dbgFile << "------------------------------------------------------------\n";

    string sLog = strprintf("\n\nSolutions (%zu):\n", vHostSolutions.size());
    sLog.reserve(1024);
    m_dbgFile << sLog;
    size_t i = 1;
    for (const auto& solution : vHostSolutions)
    {
        sLog = strprintf("Solution #%zu: ", i++);
        for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
            sLog += strprintf("%u ", solution.indices[i]);
        sLog += "\n";
        m_dbgFile << sLog;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::debugTraceSolution(const uint32_t bucketIdx)
{
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    v_uint32 vBucketHashCounters(EquihashType::NBucketCount, 0);
    copyToHost(vBucketHashCounters.data(), bucketHashCounters.get() + EquihashType::WK * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));
    const uint32_t startIdx = bucketIdx * EquihashType::NBucketSize;
    const uint32_t endIdx = startIdx + vBucketHashCounters[bucketIdx];
    
    uint32_t indices[EquihashType::ProofSize];
    uint32_t indicesNew[EquihashType::ProofSize];
    uint32_t *pIndices = indices;
    uint32_t *pIndicesNew = indicesNew;

    v_uint32 vHashes(EquihashType::NHashStorageWords, 0);
    copyToHost(vHashes.data(), hashes.get(), EquihashType::NHashStorageWords * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount, 0);

    v_uint32 vCollisionPairs(MaxCollisionsPerBucket * EquihashType::NBucketCount, 0);
    copyToHost(vCollisionPairs.data(), collisionPairs.get(), MaxCollisionsPerBucket * EquihashType::NBucketCount * sizeof(uint32_t));   

    m_dbgFile << strprintf("\n\nTracing solution for bucket #%u\n", bucketIdx);

    for (uint32_t idx = startIdx; idx < endIdx; ++idx)
    {
        const uint32_t hashIdx = idx * EquihashType::HashWords;        
        const uint32_t lastHashWord = vHashes[hashIdx + EquihashType::HashWords - 1];

        if (lastHashWord != 0)
            continue;

        m_dbgFile << strprintf("Found solution [%u-%u] \n", bucketIdx, idx);

        indices[0] = idx;
        uint32_t numIndices = 1;

        for (int round = EquihashType::WK - 1; round >= 0; --round)
        {
            copyToHost(vBucketHashIndices.data(), bucketHashIndices.get() + (round + 1) * EquihashType::NHashStorageCount,
                EquihashType::NHashStorageCount * sizeof(uint32_t));

            m_dbgFile << strprintf("Round #%u (%u indices):\n", round, numIndices);
            uint32_t numIndicesNew = 0;
            for (uint32_t index = 0; index < numIndices; ++index)
            {
                // pointer to the collision pair format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // B = bucket index, N = collision pair index
                const auto ptr = vBucketHashIndices[pIndices[index]];
                const auto collisionPairIndex = ptr & 0xFFFF;
                const auto collisionPairBucketIdx = ptr >> 16;

                const uint32_t collisionPairInfo = vCollisionPairs[collisionPairBucketIdx * MaxCollisionsPerBucket + collisionPairIndex];
                const uint32_t pairIdx1 = collisionPairInfo >> 16;
                const uint32_t pairIdx2 = collisionPairInfo & 0xFFFF;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx1;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx2;

                m_dbgFile << strprintf("#%u: %u -> [%u-%u->%u, p1: %u, p2: %u] -> new indices [%u, %u]\n", 
                    index, pIndices[index], 
                    collisionPairBucketIdx, collisionPairIndex,
                    collisionPairBucketIdx * MaxCollisionsPerBucket + collisionPairIndex,
                    pairIdx1, pairIdx2,
                    collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx1,
                    collisionPairBucketIdx * EquihashType::NBucketSize + pairIdx2);
            }
            uint32_t *pIndicesTemp = pIndices;
            pIndices = pIndicesNew;
            pIndicesNew = pIndicesTemp;
            numIndices = numIndicesNew;
            m_dbgFile << endl;
        }
        
        // map to the original indices
        m_dbgFile << "Solution final step:" << endl;
        for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
        {
            const uint32_t idx = vBucketHashIndices[pIndices[i]];
            const uint32_t origBucketIdx = idx >> 16;
            const uint32_t origIdx = idx & 0xFFFF;
            m_dbgFile << strprintf("Index %u: %u -> [%u: %u-%u] -> %u\n", 
                i, pIndices[i], idx, origBucketIdx, origIdx,
                origBucketIdx * EquihashType::NBucketSize + origIdx);
        }
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
    DEBUG_FN(debugPrintHashes(true));
    DBG_EQUI_WRITE_FN(debugWriteHashes(true));
    DBG_EQUI_WRITE_FN(debugWriteBucketIndices());

    // Perform K rounds of collision detection and XORing
    while (round < EquihashType::WK)
    {
        // Detect collisions and XOR the colliding hashes
        EQUI_TIMER_START;
        processCollisions();
        EQUI_TIMER_STOP(strprintf("Round [%u], collisions", round));
        DEBUG_FN(debugPrintCollisionPairs());
        DBG_EQUI_WRITE_FN(debugWriteCollisionPairs());
        // Swap the hash pointers for the next round
        swap(hashes, xoredHashes);
        DEBUG_FN(debugPrintHashes(false));
        DBG_EQUI_WRITE_FN(debugWriteHashes(false));

        ++round;
        cout << "Round #" << dec << round << " completed" << endl;
        DBG_EQUI_WRITE_FN(debugWriteBucketIndices());
    }

    uint32_t nSolutionCount = findSolutions();
    return nSolutionCount;
}

// Explicit template instantiation
template class EhDevice<EquihashSolver<200, 9>>;
