
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
#include <netinet/in.h>

#include <tinyformat.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <src/kernel/kernel.h>
#include <src/kernel/memutils.h>
#include <src/kernel/blake2b_device.h>
#include <src/equihash/equihash.h>
#include <src/equihash/equihash-helper.h>

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
__constant__ uint32_t CudaEquihashConstants<EquihashType>::d_HashWordOffsets[EquihashType::WK];

template<typename EquihashType>
__constant__ uint32_t CudaEquihashConstants<EquihashType>::d_HashBitOffsets[EquihashType::WK];

template<typename EquihashType>
__constant__ uint64_t CudaEquihashConstants<EquihashType>::d_HashCollisionMasks[EquihashType::WK];

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

__forceinline__ __device__ bool atomicCheckAndIncrement(uint32_t* address, const uint32_t limit, uint32_t *oldValue)
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

    uint8_t hash[BLAKE2B_OUTBYTES];
    uint32_t i = 0;
    while ((hashInputIdx < EquihashType::Base) && (i++ < numHashCalls))
    {
        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashInputIdx), sizeof(hashInputIdx));
        blake2b_final_device(&localState, hash, BLAKE2B_OUTBYTES);

        uint32_t curHashIdx = hashInputIdx * EquihashType::IndicesPerHashOutput;
        uint32_t hashByteOffset = 0;
        for (uint32_t j = 0; j < EquihashType::IndicesPerHashOutput; ++j)
        {
            // map the output hash index to the appropriate bucket
            // and store the hash in the corresponding bucket
            // index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
            //   B = bucket index, N = hash index
            const uint16_t bucketIdx = 
                (static_cast<uint16_t>(hash[hashByteOffset + 1]) << 8 |
                                       hash[hashByteOffset]) & EquihashType::NBucketIdxMask;
            
            uint32_t hashIdxInBucket = 0;
            if (!atomicCheckAndIncrement(&bucketHashCounters[bucketIdx], EquihashType::NBucketSizeExtra - 1, &hashIdxInBucket))
            {
                atomicAdd(discardedCounter, 1);
                continue;
            }
            // find the place where to store the hash (extra space exists in each bucket)
            const uint32_t bucketStorageHashIdx = bucketIdx * EquihashType::NBucketSizeExtra + hashIdxInBucket;
            const uint32_t bucketStorageHashIdxPtr = bucketStorageHashIdx * EquihashType::HashWords;

            bucketHashIndices[bucketStorageHashIdx] = curHashIdx;
            memcpy(hashes + bucketStorageHashIdxPtr, hash + hashByteOffset, EquihashType::SingleHashOutput);
            hashByteOffset += EquihashType::SingleHashOutput;
            ++curHashIdx;
        }
        ++hashInputIdx;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::generateInitialHashes()
{
    const uint32_t numHashCallsPerThread = EquihashType::NBucketSize / 8;
    const uint32_t numThreads = (EquihashType::Base + numHashCallsPerThread - 1) / numHashCallsPerThread;
    
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
    copyToHost(vBucketHashCounters.data(), 
        bucketHashCounters.get() + round * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount);
    copyToHost(vBucketHashIndices.data(), bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
        EquihashType::NHashStorageCount * sizeof(uint32_t));

    uint32_t nMaxCount = 0;
    m_dbgFile << "------\n";
    m_dbgFile << endl << "Bucket hash indices for round #" << round << ":" << endl;
    for (size_t i = 0; i < EquihashType::NBucketCount; ++i)
    {
        if (vBucketHashCounters[i] == 0)
            continue;
        if (vBucketHashCounters[i] > nMaxCount)
            nMaxCount = vBucketHashCounters[i];
        m_dbgFile << strprintf("\nRound %u, Bucket #%u, %u hash indices: ", round, i, vBucketHashCounters[i]);
        for (size_t j = 0; j < vBucketHashCounters[i]; ++j)
        {
            const uint32_t bucketHashIdx = vBucketHashIndices[i * EquihashType::NBucketSizeExtra + j];
            if (j % 20 == 0)
                m_dbgFile << endl << "#" << dec << j << ": ";
            if (round == 0)
                m_dbgFile << bucketHashIdx << " ";
            else
            {
                const uint32_t bucketIdx = bucketHashIdx >> 16;
                const uint32_t hashIdxInBucket = bucketHashIdx & 0xFFFF;
                m_dbgFile << strprintf("(%u-%u) ", bucketIdx, hashIdxInBucket);
            }
        }
        m_dbgFile << endl;
    }
    m_dbgFile << "\nRound " << dec << round << " max hash indices per bucket: " << dec << nMaxCount << endl;
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
__forceinline__ __device__ uint2 getHashIndex(const uint32_t bucketIdx, const uint32_t hashIdx, const uint32_t *bucketHashIndicesPrevPtr)
{
    const auto ptr = bucketHashIndicesPrevPtr[bucketIdx * EquihashType::NBucketSizeExtra + hashIdx];
    return make_uint2(ptr >> 16, ptr & 0xFFFF);
}

__forceinline__ __device__ bool haveDistinctCollisionIndices(const uint32_t idx1, const uint32_t idx2,
    const uint32_t *collisionPairsBucketPtr)
{
    const auto collisionPairIdx1 = collisionPairsBucketPtr[idx1];
    const auto collisionPairIdx2 = collisionPairsBucketPtr[idx2];
    const uint32_t p1 = collisionPairIdx1 >> 16;
    const uint32_t p2 = collisionPairIdx1 & 0xFFFF;
    const uint32_t p3 = collisionPairIdx2 >> 16;
    const uint32_t p4 = collisionPairIdx2 & 0xFFFF;
    return (p1 != p3) && (p1 != p4) && (p2 != p3) && (p2 != p4);
}

template <typename EquihashType>
__device__ int compare_hashes(const uint32_t* x, const uint32_t* y)
{
    uint64_t x64 = (static_cast<uint64_t>(x[1]) << 32) | x[0];
    uint64_t y64 = (static_cast<uint64_t>(y[1]) << 32) | y[0];
    uint32_t curWordOffset = 0; 
    for (size_t round = 0; round < EquihashType::WK; ++round)
    {
        if (EquihashType::HashWordOffsets[round] > curWordOffset)
        {
            curWordOffset = EquihashType::HashWordOffsets[round];
            x64 = (static_cast<uint64_t>(x[curWordOffset + 1]) << 32) | x[curWordOffset];
            y64 = (static_cast<uint64_t>(y[curWordOffset + 1]) << 32) | y[curWordOffset];
        }
        const uint32_t rx = x64 & EquihashType::HashCollisionMasks[round];
        const uint32_t ry = y64 & EquihashType::HashCollisionMasks[round];

        if (rx != rx)
        {
            // Return the difference of the first differing bits, ensuring the sign is correct
            return (rx > ry) ? 1 : -1;
        }
    }
    return 0; // Hashes are equivalent
}

template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* bucketHashIndices,
    uint32_t* bucketHashCounters,
    uint32_t* collisionPairs, 
    const uint32_t* collisionOffsets,
    uint32_t* collisionCounters,
    uint32_t* discardedCounter,
    const uint32_t round,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t wordOffset, const uint64_t collisionBitMask,
    const uint32_t xoredBitOffset, const uint32_t xoredWordOffset)
{
    const uint32_t bucketIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const auto bucketHashCountersPrevPtr = bucketHashCounters + round * EquihashType::NBucketCount;
    const auto collisionPairsPtr = collisionPairs + bucketIdx * maxCollisionsPerBucket;
    const auto collisionBucketOffset = collisionOffsets[bucketIdx];
    const uint32_t startIdxStorage = bucketIdx * EquihashType::NBucketSizeExtra;
    const uint32_t hashCount = bucketHashCountersPrevPtr[bucketIdx];
    uint32_t xoredHash[EquihashType::HashWords];
    if (hashCount == 0)
        return;

    const bool bLastRound = round == EquihashType::WK - 1;
    const auto bucketHashIndicesPrevPtr = bucketHashIndices + round * EquihashType::NHashStorageCount;
    auto bucketHashIndicesPtr = bucketHashIndices + (round + 1) * EquihashType::NHashStorageCount;
    auto bucketHashCountersPtr = bucketHashCounters + (round + 1) * EquihashType::NBucketCount;
    bool processed[EquihashType::NBucketSizeExtra] = { false };
    uint32_t stackCapacity = 20;
    uint32_t *stack;
    cudaMalloc(&stack, stackCapacity * sizeof(uint32_t));
    uint32_t stackSize;

    uint32_t hashIdxLeft = startIdxStorage * EquihashType::HashWords;
    for (uint32_t leftPairIdx = 0; leftPairIdx < hashCount - 1; 
        ++leftPairIdx, hashIdxLeft += EquihashType::HashWords, stackSize = 0)
    {
        if (processed[leftPairIdx])
            continue;

        stack[stackSize++] = leftPairIdx;
        // each collision info holds up to 2 collision pairs
        // first one points to the left pair, the next 2 point to the right collision pairs
        const uint32_t hashWordIdxLeft = hashIdxLeft + wordOffset;
        const uint64_t hashLeft = 
            ((static_cast<uint64_t>(hashes[hashWordIdxLeft + 1]) << 32) | 
                                    hashes[hashWordIdxLeft]);
        const uint64_t maskedHashLeft = hashLeft & collisionBitMask;
        
        uint32_t hashIdxRight = (startIdxStorage + leftPairIdx + 1) * EquihashType::HashWords;
        for (uint32_t rightPairIdx = leftPairIdx + 1;  rightPairIdx < hashCount; 
            ++rightPairIdx, hashIdxRight += EquihashType::HashWords)
        {
            if (processed[rightPairIdx])
                continue;

            const uint32_t hashWordIdxRight = hashIdxRight + wordOffset;
            const uint64_t hashRight = (static_cast<uint64_t>(hashes[hashWordIdxRight + 1]) << 32) | 
                                                              hashes[hashWordIdxRight];
            const uint64_t maskedHashRight = hashRight & collisionBitMask;
            if (maskedHashLeft == maskedHashRight)
            {
                processed[rightPairIdx] = true;
                for (uint32_t i = 0; i < stackSize; ++i)
                {
                    // hash collision found - xor the hashes and store the result
                    bool bAllZeroes = true;
                    const uint32_t hashBase = (startIdxStorage + stack[i]) * EquihashType::HashWords;
                    for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                    {
                        xoredHash[j] = hashes[hashBase + j] ^ hashes[hashIdxRight + j];
                        if (xoredHash[j])
                            bAllZeroes = false;
                    }
                    // accept all zeroes hash result at the last round
                    if (bAllZeroes && !bLastRound)
                        continue; // skip if all zeroes

                    if (round > 0)
                    {
                        // skip this collision if it is based on the hash pair from the same bucket and with repeated previous collision indices
                        const auto prevHashIdx1 = getHashIndex<EquihashType>(bucketIdx, stack[i], bucketHashIndicesPrevPtr);
                        const auto prevHashIdx2 = getHashIndex<EquihashType>(bucketIdx, rightPairIdx, bucketHashIndicesPrevPtr);
                        if ((prevHashIdx1.x == prevHashIdx2.x) && 
                            !haveDistinctCollisionIndices(prevHashIdx1.y, prevHashIdx2.y, collisionPairs + prevHashIdx1.x * maxCollisionsPerBucket))
                            continue;
                    }

                    // define xored hash bucket based on the first NBucketIdxMask bits (starting from the CollisionBitLength)                
                    uint32_t xoredBucketIdx = 
                        (static_cast<uint32_t>(((static_cast<uint64_t>(xoredHash[xoredWordOffset + 1]) << 32) | 
                                                                       xoredHash[xoredWordOffset]) >> xoredBitOffset));
                    if (round % 2 == 0)
                        xoredBucketIdx = (xoredBucketIdx >> 4) | (xoredBucketIdx & 0x0F);
                    xoredBucketIdx &= EquihashType::NBucketIdxMask;
                    uint32_t xoredHashIdxInBucket = 0;
                    if (bLastRound)
                    {
                        if (xoredBucketIdx != 0)
                            continue; // skip if the bucket is not zero

                        xoredHashIdxInBucket = atomicAdd(&bucketHashCountersPtr[0], 1);
                    }
                    else if (!atomicCheckAndIncrement(&bucketHashCountersPtr[xoredBucketIdx], EquihashType::NBucketSizeExtra - 1, &xoredHashIdxInBucket))
                    {
                        atomicAdd(discardedCounter, 1);
                        continue; // skip if the bucket is full
                    }
                    const uint32_t xoredBucketHashIdxStorage = xoredBucketIdx * EquihashType::NBucketSizeExtra + xoredHashIdxInBucket;
                    const uint32_t xoredBucketHashIdxStoragePtr = xoredBucketHashIdxStorage * EquihashType::HashWords;
                    for (uint32_t j = 0; j < EquihashType::HashWords; ++j)
                        xoredHashes[xoredBucketHashIdxStoragePtr + j] = xoredHash[j];
                    const uint32_t collisionPairIdx = collisionBucketOffset + collisionCounters[bucketIdx];

                    // hash index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                    // B = bucket index, N = collision pair index
                    bucketHashIndicesPtr[xoredBucketHashIdxStorage] = (bucketIdx << 16) | collisionPairIdx;
                    if (compare_hashes<EquihashType>(hashes + hashWordIdxLeft, hashes + hashWordIdxRight) < 0)
                       collisionPairsPtr[collisionPairIdx] = (rightPairIdx << 16) | stack[i];
                    else
                        collisionPairsPtr[collisionPairIdx] = (stack[i] << 16) | rightPairIdx;
                    collisionCounters[bucketIdx] += 1;
                }
                stack[stackSize++] = rightPairIdx;
                if (stackSize >= stackCapacity)
                {
                    stackCapacity *= 2;
                    uint32_t *newStack;
                    cudaMalloc(&newStack, stackCapacity * sizeof(uint32_t));
                    memcpy(newStack, stack, stackSize * sizeof(uint32_t));
                    cudaFree(stack);
                    stack = newStack;
                }
            }
        }
    }
}

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
    cudaMemset(discardedCounter.get(), 0, sizeof(uint32_t));

    dim3 gridDim((EquihashType::NBucketCount + ThreadsPerBlock - 1) / ThreadsPerBlock);
    dim3 blockDim(ThreadsPerBlock);

    try {
        cudaMemset(collisionCounters.get(), 0, EquihashType::NBucketCount * sizeof(uint32_t));

        cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                    hashes.get(), xoredHashes.get(),
                    bucketHashIndices.get(),
                    bucketHashCounters.get(),
                    collisionPairs.get(),
                    collisionOffsets.get() + round * EquihashType::NBucketCount, 
                    collisionCounters.get(),
                    discardedCounter.get(),
                    round,
                    MaxCollisionsPerBucket,
                    EquihashType::HashWordOffsets[round],
                    EquihashType::HashCollisionMasks[round],
                    EquihashType::XoredHashBitOffsets[round],
                    EquihashType::XoredHashWordOffsets[round]
                );

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
    const auto bucketHashCountersLast = bucketHashCounters + EquihashType::WK * EquihashType::NBucketCount;
    const uint32_t hashCount = bucketHashCountersLast[0];

    uint32_t indices[EquihashType::ProofSize];
    uint32_t indicesNew[EquihashType::ProofSize];
    uint32_t *pIndices = indices;
    uint32_t *pIndicesNew = indicesNew;
    uint32_t hashIdx = 0;
    uint32_t nSolutionCount = 0;

    const auto bucketHashIndicesLastPtr = bucketHashIndices + EquihashType::WK * EquihashType::NHashStorageCount;

    for (uint32_t mainIndex = 0; mainIndex < hashCount; 
        ++mainIndex, hashIdx += EquihashType::HashWords)
    {
        auto hashPtr = hashes + hashIdx + EquihashType::HashWords - 2;
        if (hashPtr[0] || hashPtr[1])
            continue;

        pIndices[0] = mainIndex;
        uint32_t numIndices = 1;

        bool bDistinctIndices = true;
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
                const auto collisionPairBucketIdxMasked = ptr & 0xFFFF0000;
                const auto collisionPairBucketIdx = ptr >> 16;
                const auto collisionPairIndex = ptr & 0xFFFF;

                const auto collisionPairsPtr = collisionPairs + collisionPairBucketIdx * maxCollisionsPerBucket;
                const uint32_t collisionPairInfo = collisionPairsPtr[collisionPairIndex];
                const uint32_t pairIdx1 = collisionPairInfo >> 16;
                const uint32_t pairIdx2 = collisionPairInfo & 0xFFFF;
                
                uint32_t newIndex = collisionPairBucketIdxMasked | pairIdx1;
                for (uint32_t i = 0; i < numIndicesNew; ++i)
                {
                    if (pIndicesNew[i] == newIndex)
                    {
                        bDistinctIndices = false;
                        break;
                    }
                }
                if (!bDistinctIndices)
                    break;
                pIndicesNew[numIndicesNew++] = newIndex;
                newIndex = collisionPairBucketIdxMasked | pairIdx2;
                for (uint32_t i = 0; i < numIndicesNew; ++i)
                {
                    if (pIndicesNew[i] == newIndex)
                    {
                        bDistinctIndices = false;
                        break;
                    }
                }
                if (!bDistinctIndices)
                    break;
                pIndicesNew[numIndicesNew++] = newIndex;
            }
            if (!bDistinctIndices)
                break;
            uint32_t *pIndicesTemp = pIndices;
            pIndices = pIndicesNew;
            pIndicesNew = pIndicesTemp;
            numIndices = numIndicesNew;
        }
        
        if (!bDistinctIndices)
            continue;

        // found solution
        printf("Found solution [%u] \n", mainIndex);

        // map to the original indices
        for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
        {
            const auto idx = pIndices[i];
            const auto storageIdx = (idx >> 16) * EquihashType::NBucketSizeExtra + (idx & 0xFFFF);
            solutions[nSolutionCount].indices[i] = bucketHashIndices[storageIdx];
        }
        if (++nSolutionCount >= maxSolutionCount)
            break;
    }
    *solutionCount = nSolutionCount;
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::findSolutions()
{
    uint32_t numSolutions = 0;
    cudaMemset(solutionCount.get(), 0, sizeof(uint32_t));

    cudaKernel_findSolutions<EquihashType><<<1, 1>>>(
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

    m_dbgFile << "------\n";
    v_uint32 vBucketHashCounters(EquihashType::NBucketCount, 0);
    copyToHost(vBucketHashCounters.data(), bucketHashCounters.get() + (bInitialHashes ? 0 : (round + 1) * EquihashType::NBucketCount),
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount, 0);
    copyToHost(vBucketHashIndices.data(), bucketHashIndices.get() + (bInitialHashes ? 0 : (round + 1) * EquihashType::NHashStorageCount),
        vBucketHashIndices.size() * sizeof(uint32_t));
    
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
        if (bInitialHashes)
            sLog = strprintf("\nInitial bucket #%u, (%u) hashes:\n", bucketIdx, nBucketHashCount);
        else
            sLog = strprintf("\nRound %u bucket #%u, (%u) hashes:\n", round, bucketIdx, nBucketHashCount);
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
                sLog += strprintf("%08x ", htonl(vHostHashes[hashInputIdx]));
            }
            sLog += strprintf("| %u", vBucketHashIndices[bucketHashStorageIdx + i]);
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
        if (bInitialHashes)
        cout << strprintf("\nInitial bucket #%u, (%u) hashes:\n", bucketIdx, nBucketHashCount);
        else
            cout << strprintf("\nRound %u Bucket #%u, (%u) hashes:\n", round, bucketIdx, nBucketHashCount);

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
                sLog += strprintf("%08x ", htonl(vHostHashes[hashInputIdx]));
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
    v_uint32 vCollPairsOffsets(EquihashType::NBucketCount, 0);
    copyToHost(vBucketCollisionCounts.data(),
        collisionCounters.get(), EquihashType::NBucketCount * sizeof(uint32_t));

    copyToHost(vCollPairsOffsets.data(),
        collisionOffsets.get() + round * EquihashType::NBucketCount,
        vCollPairsOffsets.size() * sizeof(uint32_t));

    v_uint32 vCollisionPairs(EquihashType::NBucketCount * MaxCollisionsPerBucket);
    copyToHost(vCollisionPairs.data(), 
        collisionPairs.get(), vCollisionPairs.size() * sizeof(uint32_t));

    constexpr uint32_t COLLISIONS_PER_LINE = 10;
    m_dbgFile << "------\n";
    m_dbgFile << endl << "Collision pairs for round #" << round << ":" << endl;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        size_t nBucketCollisionCount = vBucketCollisionCounts[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
        m_dbgFile << strprintf("\nRound %u, Bucket #%u, collision pairs %u (bucket offsets: %u...%u), collision bucket origin: %u:\n",
            round, bucketIdx, nBucketCollisionCount, vCollPairsOffsets[bucketIdx], vCollPairsOffsets[bucketIdx] + nBucketCollisionCount,
            bucketIdx * MaxCollisionsPerBucket);

        for (uint32_t i = 0; i < nBucketCollisionCount; ++i)
        {
            const uint32_t collisionPairInfo = vCollisionPairs[bucketIdx * MaxCollisionsPerBucket +
                vCollPairsOffsets[bucketIdx] + i];
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
    m_dbgFile << "------\n";

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
    const uint32_t startIdx = bucketIdx * EquihashType::NBucketSizeExtra;
    
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

    uint32_t hashIdx = startIdx * EquihashType::HashWords;
    for (uint32_t idxInBucket = 0; idxInBucket < vBucketHashCounters[bucketIdx]; 
        ++idxInBucket, hashIdx += EquihashType::HashWords)
    {
        const uint32_t lastHashWord = vHashes[hashIdx + EquihashType::HashWords - 1];

        if (lastHashWord != 0)
            continue;

        m_dbgFile << strprintf("Found solution [bucket %u-%u]\n",
            bucketIdx, idxInBucket);

        indices[0] = bucketIdx * EquihashType::NBucketSizeExtra + idxInBucket;
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
                const auto collisionPairBucketIdx = ptr >> 16;
                const auto collisionPairIndex = ptr & 0xFFFF;

                const uint32_t collisionPairInfo = vCollisionPairs[collisionPairBucketIdx * MaxCollisionsPerBucket + collisionPairIndex];
                const uint32_t pairIdx1 = collisionPairInfo >> 16;
                const uint32_t pairIdx2 = collisionPairInfo & 0xFFFF;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSizeExtra + pairIdx1;
                pIndicesNew[numIndicesNew++] = collisionPairBucketIdx * EquihashType::NBucketSizeExtra + pairIdx2;

                m_dbgFile << strprintf("#%u: %u -> [collision #%u-%u->%u, p1:%u, p2:%u] -> new indices [%u, %u]\n", 
                    index, pIndices[index], 
                    collisionPairBucketIdx, collisionPairIndex,
                    collisionPairBucketIdx * MaxCollisionsPerBucket + collisionPairIndex,
                    pairIdx1, pairIdx2,
                    collisionPairBucketIdx * EquihashType::NBucketSizeExtra + pairIdx1,
                    collisionPairBucketIdx * EquihashType::NBucketSizeExtra + pairIdx2);
            }
            uint32_t *pIndicesTemp = pIndices;
            pIndices = pIndicesNew;
            pIndicesNew = pIndicesTemp;
            numIndices = numIndicesNew;
            m_dbgFile << endl;
        }
        
        copyToHost(vBucketHashIndices.data(), bucketHashIndices.get(),
        EquihashType::NHashStorageCount * sizeof(uint32_t));

        // map to the original indices
        v_uint32 vSolution(EquihashType::ProofSize, 0);
        m_dbgFile << "Solution final step:" << endl;
        for (uint32_t i = 0; i < EquihashType::ProofSize; ++i)
        {
            vSolution[i] = vBucketHashIndices[pIndices[i]];
            m_dbgFile << strprintf("Index %u: %u -> %u\n", i, pIndices[i], vSolution[i]);
        }

        blake2b_state currState;
        copyToHost(&currState, initialState.get(), sizeof(blake2b_state));

        string sError;
        auto eh = EquihashSolver<EquihashType::WN, EquihashType::WK>();
        // check the solution
        v_uint8 solutionMinimal = GetMinimalFromIndices(vSolution, EquihashType::CollisionBitLength);
        if (!eh.IsValidSolution(sError, currState, solutionMinimal))
            m_dbgFile << endl << sError << endl;
        else
            m_dbgFile << endl << "Valid solution" << endl;
        string sHexSolution = HexStr(solutionMinimal);
        m_dbgFile << "Solution(" << dec << solutionMinimal.size() << "):" 
            << endl << sHexSolution << endl;
        break;
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

    return findSolutions();
}

// Explicit template instantiation
template class EhDevice<EquihashSolver<200, 9>>;
