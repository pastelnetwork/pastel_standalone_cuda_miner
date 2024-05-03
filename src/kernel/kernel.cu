
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

template <typename EquihashType>
__constant__ uint32_t HashWordOffsets[EquihashType::WK];

template <typename EquihashType>
__constant__ uint64_t HashCollisionMasks[EquihashType::WK];

template<typename EquihashType>
__constant__ uint32_t HashBitOffsets[EquihashType::WK];

template<typename EquihashType>
__constant__ uint32_t XoredHashBitOffsets[EquihashType::WK];

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
EhDevice<EquihashType>::EhDevice()
{
    if (!allocate_memory())
        throw runtime_error("Failed to allocate CUDA memory for Equihash solver");

    // Copy the constants to the device
    cudaMemcpyToSymbol(HashWordOffsets<EquihashType>, EquihashType::HashWordOffsets.data(), 
                    EquihashType::WK * sizeof(uint32_t));

    cudaMemcpyToSymbol(HashCollisionMasks<EquihashType>, EquihashType::HashCollisionMasks.data(),
                    EquihashType::WK * sizeof(uint64_t));

    cudaMemcpyToSymbol(HashBitOffsets<EquihashType>, EquihashType::HashBitOffsets.data(),
                    EquihashType::WK * sizeof(uint32_t));

    cudaMemcpyToSymbol(XoredHashBitOffsets<EquihashType>, EquihashType::XoredHashBitOffsets.data(),
                    EquihashType::WK * sizeof(uint32_t));
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


template<typename EquihashType>
__device__ void ExpandArray(const uint8_t* in, size_t nInBytesLen, uint32_t* out, const size_t nOutWords, size_t nBitsLength)
{
    assert(nBitsLength >= 8);
    assert(8 * sizeof(uint32_t) >= 7 + nBitsLength);

    size_t out_bytes_per_element = (nBitsLength + 7) / 8;
    assert(nOutWords == out_bytes_per_element * nInBytesLen * 8 / nBitsLength);

    uint32_t bit_len_mask = ((uint32_t)1 << nBitsLength) - 1;
    size_t acc_bits = 0;
    uint32_t acc_value = 0;
    size_t out_word_index = 0;

    for (size_t in_byte_index = 0; in_byte_index < nInBytesLen; in_byte_index++)
    {
        acc_value = (acc_value << 8) | in[in_byte_index];
        acc_bits += 8;

        if (acc_bits >= nBitsLength)
        {
            acc_bits -= nBitsLength;
            if (out_word_index < nOutWords)
                out[out_word_index++] = (acc_value >> acc_bits) & bit_len_mask;
        }
    }

    // Zero out any remaining bits in the last word
    if (EquihashType::HashPartialBytesLeft > 0 && out_word_index < nOutWords)
    {
        uint32_t mask = ((uint32_t)1 << (EquihashType::HashPartialBytesLeft * 8)) - 1;
        out[out_word_index] &= mask;
    }
}

// CUDA kernel to generate initial hashes from blake2b state
template<typename EquihashType>
__global__ void cudaKernel_generateInitialHashes(const blake2b_state* state, uint32_t* hashes, 
    uint32_t *bucketHashIndices, uint32_t *bucketHashCounters, uint32_t *discardedCounter,
    const uint32_t numHashCalls)
{
    const uint32_t tid = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    uint32_t hashInputIdx = tid * numHashCalls;

    uint8_t hash[BLAKE2B_OUTBYTES];
    uint32_t i = 0;
    while ((hashInputIdx < EquihashType::Base) && (i++ < numHashCalls))
    {
        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashInputIdx), sizeof(hashInputIdx));
        blake2b_final_device(&localState, hash, BLAKE2B_OUTBYTES);

        uint32_t curHashIdx = __umul24(hashInputIdx, EquihashType::IndicesPerHashOutput);
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
            
            const uint32_t hashIdxInBucket = atomicAdd(&bucketHashCounters[bucketIdx], 1);
            if (hashIdxInBucket >= EquihashType::NBucketSizeExtra)
            {
                atomicAdd(discardedCounter, 1);
                atomicSub(&bucketHashCounters[bucketIdx], 1);
                continue;
            }
            // find the place where to store the hash (extra space exists in each bucket)
            const uint32_t bucketStorageHashIdx = __umul24(bucketIdx, EquihashType::NBucketSizeExtra) + hashIdxInBucket;
            const uint32_t bucketStorageHashIdxPtr = __umul24(bucketStorageHashIdx, EquihashType::HashWords);

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
__forceinline__ __device__ uint2 getHashIndex(const uint32_t ptr)
{
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

// __device__ uint32_t atomicCheckAndIncrementTimeHigh = 0;
// __device__ uint32_t atomicCheckAndIncrementTimeLow = 0;

template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* bucketHashIndicesPrevPtr, // bucketHashIndices + __umul24(round, EquihashType::NHashStorageCount)
    uint32_t* bucketHashIndicesPtr, // bucketHashIndices + __umul24(round + 1, EquihashType::NHashStorageCount)
    uint32_t* bucketHashCountersPrevPtr, // bucketHashCounters + __umul24(round, EquihashType::NBucketCount)
    uint32_t* bucketHashCountersPtr, // bucketHashCounters + __umul24(round + 1, EquihashType::NBucketCount);
    uint32_t* collisionPairs, 
    const uint32_t* collisionOffsets,
    uint32_t* collisionCounters,
    uint32_t* discardedCounter,
    const uint32_t round,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t wordOffset, const uint64_t collisionBitMask,
    const uint32_t xoredBitOffset, const uint32_t xoredWordOffset)
{
    const uint32_t bucketIdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

    const auto collisionPairsPtr = collisionPairs + __umul24(bucketIdx, maxCollisionsPerBucket);
    const auto collisionBucketOffset = collisionOffsets[bucketIdx];
    const uint32_t startIdxStorage = __umul24(bucketIdx, EquihashType::NBucketSizeExtra);
    const uint32_t hashCount = bucketHashCountersPrevPtr[bucketIdx];
    uint32_t xoredHash[EquihashType::HashWords];
    if (hashCount == 0)
        return;

    const bool bLastRound = round == EquihashType::WK - 1;
    bool processed[EquihashType::NBucketSizeExtra];
    memset(processed, 0, EquihashType::NBucketSizeExtraBoolMaskSize);
    uint32_t stack[20];
    uint32_t stackSize = 0;

    //uint64_t start, stop;
    //uint64_t localAtomicTime = 0;

    uint32_t hashIdxLeft = __umul24(startIdxStorage, EquihashType::HashWords);
    for (uint32_t leftPairIdx = 0; leftPairIdx < hashCount - 1; 
        ++leftPairIdx, hashIdxLeft += EquihashType::HashWords, stackSize = 0)
    {
        if (processed[leftPairIdx])
            continue;

        stack[stackSize++] = leftPairIdx;
        const uint32_t hashWordIdxLeft = hashIdxLeft + wordOffset;
        const uint64_t maskedHashLeft = 
            ((static_cast<uint64_t>(hashes[hashWordIdxLeft + 1]) << 32) | 
                                    hashes[hashWordIdxLeft]) & collisionBitMask;
        
        uint32_t hashIdxRight = __umul24(startIdxStorage + leftPairIdx + 1, EquihashType::HashWords);
        for (uint32_t rightPairIdx = leftPairIdx + 1;  rightPairIdx < hashCount; 
            ++rightPairIdx, hashIdxRight += EquihashType::HashWords)
        {
            if (processed[rightPairIdx])
                continue;

            const uint32_t hashWordIdxRight = hashIdxRight + wordOffset;
            const uint64_t maskedHashRight = 
                ((static_cast<uint64_t>(hashes[hashWordIdxRight + 1]) << 32) | 
                                        hashes[hashWordIdxRight]) & collisionBitMask;
            if (maskedHashLeft != maskedHashRight)
                continue;

            processed[rightPairIdx] = true;
            for (uint32_t i = 0; i < stackSize; ++i)
            {
                // hash collision found - xor the hashes and store the result
                const uint32_t hashBase = __umul24(startIdxStorage + stack[i], EquihashType::HashWords);
                for (uint32_t j = wordOffset; j < EquihashType::HashWords; ++j)
                    xoredHash[j] = hashes[hashBase + j] ^ hashes[hashIdxRight + j];
                bool bAllZeroes = true;
                for (uint32_t j = wordOffset; j < EquihashType::HashWords; ++j)
                {
                    if (__popc(xoredHash[j]) > 0)
                    {
                        bAllZeroes = false;
                        break;
                    }
                }
                // accept all zeroes hash result at the last round
                if (bAllZeroes && !bLastRound)
                    continue; // skip if all zeroes

                if (round > 0)
                {
                    // skip this collision if it is based on the hash pair from the same bucket and with repeated previous collision indices
                    const auto prevHashIdx1 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[startIdxStorage + stack[i]]);
                    const auto prevHashIdx2 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[startIdxStorage + rightPairIdx]);
                    if ((prevHashIdx1.x == prevHashIdx2.x) && 
                        !haveDistinctCollisionIndices(prevHashIdx1.y, prevHashIdx2.y, 
                            collisionPairs + __umul24(prevHashIdx1.x, maxCollisionsPerBucket)))
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
                else {
                    xoredHashIdxInBucket = atomicAdd(&bucketHashCountersPtr[xoredBucketIdx], 1);
                    if (xoredHashIdxInBucket >= EquihashType::NBucketSizeExtra)
                    {
                        atomicAdd(discardedCounter, 1);
                        atomicSub(&bucketHashCountersPtr[xoredBucketIdx], 1);
                        continue; // skip if the bucket is full
                    }
                }
                const uint32_t xoredBucketHashIdxStorage = 
                    __umul24(xoredBucketIdx, EquihashType::NBucketSizeExtra) + xoredHashIdxInBucket;
                const uint32_t xoredBucketHashIdxStoragePtr = 
                    __umul24(xoredBucketHashIdxStorage, EquihashType::HashWords);
                memcpy(xoredHashes + xoredBucketHashIdxStoragePtr + wordOffset, xoredHash + wordOffset,
                    __umul24(EquihashType::HashWords - wordOffset, sizeof(uint32_t)));
                const uint32_t collisionPairIdx = collisionBucketOffset + collisionCounters[bucketIdx];

                // hash index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // B = bucket index, N = collision pair index
                bucketHashIndicesPtr[xoredBucketHashIdxStorage] = (bucketIdx << 16) | collisionPairIdx;
                collisionPairsPtr[collisionPairIdx] = (rightPairIdx << 16) | stack[i];
                ++(collisionCounters[bucketIdx]);
            }
            if (stackSize >= 20)
                break;
            stack[stackSize++] = rightPairIdx;
        }
    }
    //atomicAdd(&atomicCheckAndIncrementTimeHigh, static_cast<uint32_t>(localAtomicTime >> 32));
    //atomicAdd(&atomicCheckAndIncrementTimeLow, static_cast<uint32_t>(localAtomicTime));
}

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
    cudaMemset(discardedCounter.get(), 0, sizeof(uint32_t));

    dim3 gridDim((EquihashType::NBucketCount + EquihashType::CollisionThreadsPerBlock - 1) / EquihashType::CollisionThreadsPerBlock);
    dim3 blockDim(EquihashType::CollisionThreadsPerBlock);

    try {
        cudaMemset(collisionCounters.get(), 0, EquihashType::NBucketCount * sizeof(uint32_t));
        // uint32_t resetValue = 0;
        // cudaMemcpyToSymbol(atomicCheckAndIncrementTimeHigh, &resetValue, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
        // cudaMemcpyToSymbol(atomicCheckAndIncrementTimeLow, &resetValue, sizeof(uint32_t), 0, cudaMemcpyHostToDevice);

        cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                    hashes.get(), xoredHashes.get(),
                    bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
                    bucketHashIndices.get() + (round + 1) * EquihashType::NHashStorageCount,
                    bucketHashCounters.get() + round * EquihashType::NBucketCount,
                    bucketHashCounters.get() + (round + 1) * EquihashType::NBucketCount,
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
        copyToHost(vCollisionCounters.data(), collisionCounters.get(), EquihashType::NBucketCountStorageSize);

        // Store the accumulated collision pair offset for the current round
        for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
            vCollisionPairsOffsets[bucketIdx] += vCollisionCounters[bucketIdx];
        
        copyToDevice(collisionOffsets.get() + (round + 1) * EquihashType::NBucketCount, vCollisionPairsOffsets.data(),
            EquihashType::NBucketCountStorageSize);

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
            const auto bucketHashIndicesRoundPtr = bucketHashIndices + __umul24(round + 1, EquihashType::NHashStorageCount);

            uint32_t numIndicesNew = 0;
            for (uint32_t index = 0; index < numIndices; ++index)
            {
                // pointer to the collision pair format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
                // B = bucket index, N = collision pair index
                const auto idx = pIndices[index];
                const auto idxBucket = idx >> 16;
                const auto idxInBucket = idx & 0xFFFF;

                const auto storageIdx = __umul24(idxBucket, EquihashType::NBucketSizeExtra) + idxInBucket;
                const auto ptr = bucketHashIndicesRoundPtr[storageIdx];
                const auto collisionPairBucketIdxMasked = ptr & 0xFFFF0000;
                const auto collisionPairBucketIdx = ptr >> 16;
                const auto collisionPairIndex = ptr & 0xFFFF;

                const auto collisionPairsPtr = collisionPairs + __umul24(collisionPairBucketIdx, maxCollisionsPerBucket);
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
        for (uint32_t i = 0; i < EquihashType::ProofSize; i += 2)
        {
            const auto idx1 = pIndices[i];
            const auto storageIdx1 = __umul24(idx1 >> 16, EquihashType::NBucketSizeExtra) + (idx1 & 0xFFFF);
            const auto newIndex1 = bucketHashIndices[storageIdx1];
            const auto idx2 = pIndices[i + 1];
            const auto storageIdx2 = __umul24(idx2 >> 16, EquihashType::NBucketSizeExtra) + (idx2 & 0xFFFF);
            const auto newIndex2 = bucketHashIndices[storageIdx2];
            if (newIndex1 < newIndex2)
            {
                solutions[nSolutionCount].indices[i] = newIndex1;
                solutions[nSolutionCount].indices[i + 1] = newIndex2;
            } else {
                solutions[nSolutionCount].indices[i] = newIndex2;
                solutions[nSolutionCount].indices[i + 1] = newIndex1;
            }
        }

        // sort the indices in the solution
        // indices in each group should be sorted so that the first index in one group is 
        // less than the first index in the next group
        for (uint32_t groupSize = 2; groupSize < EquihashType::ProofSize; groupSize *= 2)
        {
            for (uint32_t i = 0; i < EquihashType::ProofSize; i += groupSize*2)
            {
                if (solutions[nSolutionCount].indices[i] < solutions[nSolutionCount].indices[i + groupSize])
                    continue;
                for (uint32_t j = i; j < i + groupSize; ++j)
                {
                    const uint32_t temp = solutions[nSolutionCount].indices[j];
                    solutions[nSolutionCount].indices[j] = solutions[nSolutionCount].indices[j + groupSize];
                    solutions[nSolutionCount].indices[j + groupSize] = temp;
                }
            }
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
void EhDevice<EquihashType>::copySolutionsToHost(vector<typename EquihashType::solution_type> &vHostSolutions)
{
    uint32_t nSolutionCount = 0;
    copyToHost(&nSolutionCount, solutionCount.get(), sizeof(uint32_t));

    // Resize the host solutions vector
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), solutions.get(), nSolutionCount * sizeof(typename EquihashType::solution_type));
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
        // uint64_t totalAtomicTimeHigh = 0;
        // uint64_t totalAtomicTimeLow = 0;
        EQUI_TIMER_START;
        processCollisions();
        // cudaMemcpyFromSymbol(&totalAtomicTimeHigh, atomicCheckAndIncrementTimeHigh, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
        // cudaMemcpyFromSymbol(&totalAtomicTimeLow, atomicCheckAndIncrementTimeLow, sizeof(uint32_t), 0, cudaMemcpyDeviceToHost);
        //uint64_t totalAtomicTime = (static_cast<uint64_t>(totalAtomicTimeHigh) << 32) | totalAtomicTimeLow;
        //double atomicTimeMs = static_cast<double>(totalAtomicTime) / (1000.0 * 1000.0);
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

    EQUI_TIMER_START;
    uint32_t nSolutionCount = findSolutions();
    EQUI_TIMER_STOP("Solution search");
    return nSolutionCount;
}

// Explicit template instantiation
template class EhDevice<EquihashSolver<200, 9>>;
