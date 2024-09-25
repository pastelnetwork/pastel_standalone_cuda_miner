
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

#ifdef __linux__
#include <netinet/in.h>
#endif

#include <tinyformat.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <src/kernel/kernel.h>
#include <src/kernel/blake2b_device.h>
#include <src/kernel/byteswap.h>
#include <src/equihash/equihash.h>
#include <src/equihash/equihash-helper.h>
#include <src/equihash/equihash-types.h>
#ifdef _WIN32_
#include <src/utils/logger.h>
using namespace spdlog;
#endif 

using namespace std;

template <typename EquihashType>
__constant__ uint32_t HashWordOffsets[EquihashType::WK];

template <typename EquihashType>
__constant__ uint64_t HashCollisionMasks[EquihashType::WK];

template<typename EquihashType>
__constant__ uint32_t HashBitOffsets[EquihashType::WK];

template<typename EquihashType>
__constant__ uint32_t XoredHashBitOffsets[EquihashType::WK];

union ByteToUint32
{
    uchar4 u4;
    uint32_t value;
};

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

// Get the maximum grid size supported by the device
MaxGridSize getMaxGridSize(int deviceId)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceId);
	return MaxGridSize { deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] };
}

template<typename EquihashType>
EhDevice<EquihashType>::EhDevice(bool bNewSolver) :
	bUseNewSolver(bNewSolver)
{
	nCudaDeviceCount = getNumCudaDevices();
	if (nCudaDeviceCount == 0)
		throw runtime_error("No CUDA devices found");
	int deviceId = 0;
	cudaGetDevice(&deviceId);
	nCudaMaxThreads = getMaxThreadsPerBlock(deviceId);
	CudaMaxGridSize = getMaxGridSize(deviceId);

	if (!allocate_memory())
        throw runtime_error("Failed to allocate CUDA memory for Equihash solver");

    if (!bUseNewSolver)
    {
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
}

template<typename EquihashType>
bool EhDevice<EquihashType>::allocate_memory()
{
    try
    {
        // Allocate device memory for blake2b state
        d_initialState = make_cuda_unique<blake2b_state>(1);
		const uint32_t nHashStorageWords = bUseNewSolver ? EquihashType::NHashStorageWordsEx : EquihashType::NHashStorageWords;
		// Allocate device memory for hash values
        //      new solver: 23'552'000 * 4 = 94'208'000 bytes (~90 MB)
        d_hashes = make_cuda_unique<uint32_t>(nHashStorageWords);
        d_xoredHashes = make_cuda_unique<uint32_t>(nHashStorageWords);

		// Allocate device buffer for bucket hash indices: 2'355'200 * 10 * 4 = 94'208'000 bytes (~90 MB)
        d_bucketHashIndices = make_cuda_unique<uint32_t>(EquihashType::NHashStorageCount *(EquihashType::WK + 1));
		// Allocate device buffer for bucket hash counters: 2'048 * 10 * 4 = 81'920 bytes (80 KB)
        d_bucketHashCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));

		// Allocate device buffer for collision pair pointers: 2'048 * 10'000 * 4 = 81'920'000 bytes (~78 MB)
        d_collisionPairs = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * MaxCollisionsPerBucket);
        d_collisionCounters = make_cuda_unique<uint32_t>(EquihashType::NBucketCount);
        vCollisionCounters.resize(EquihashType::NBucketCount, 0);

        d_discardedCounter = make_cuda_unique<uint32_t>(1);

		// collision pair offsets for each bucket for each round: 2'048 * 10 * 4 = 81'920 bytes (80 KB)
        d_collisionOffsets = make_cuda_unique<uint32_t>(EquihashType::NBucketCount * (EquihashType::WK + 1));
        vCollisionPairsOffsets.resize(EquihashType::NBucketCount, 0);

		// Allocate device memory for solutions and solution count: 512 * 4 * 20 = 40'960 bytes (40 KB)
        d_solutions = make_cuda_unique<typename EquihashType::solution_type>(MaxSolutions);
        d_solutionCount = make_cuda_unique<uint32_t>(1);

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        return false;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::clear()
{
	round = 0;
	bBreakSolver = false;
	cudaMemset(d_bucketHashCounters.get(), 0, EquihashType::NBucketCount * (EquihashType::WK + 1) * sizeof(uint32_t));
	cudaMemset(d_solutionCount.get(), 0, sizeof(uint32_t));
	vCollisionPairsOffsets.assign(EquihashType::NBucketCount, 0);
}

// CUDA kernel to generate initial hashes from blake2b state
template<typename EquihashType>
__global__ void cudaKernel_generateInitialHashes(const blake2b_state* state, uint32_t* hashes, 
    uint32_t *bucketHashIndices, uint32_t *bucketHashCounters, uint32_t *discardedCounter)
{
    uint32_t hashInputIdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	if (hashInputIdx >= EquihashType::Base)
		return;

    uint8_t hash[BLAKE2B_OUTBYTES];
    blake2b_state localState = *state;
	const uint32_t hashInputIdxLE = htole32_device(hashInputIdx);
    blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashInputIdxLE), sizeof(hashInputIdxLE));
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
}

// CUDA kernel to generate initial hashes from blake2b state
template<typename EquihashType>
__global__ void cudaKernel_generateInitialHashes_new(const blake2b_state* state, uint32_t* hashes, 
    uint32_t *bucketHashIndices, uint32_t *bucketHashCounters, uint32_t *discardedCounter)
{
    uint32_t hashInputIdx = threadIdx.x + __umul24(blockIdx.x, blockDim.x);
	if (hashInputIdx >= EquihashType::Base)
		return;

	uint8_t hash[BLAKE2B_OUTBYTES];
	uint32_t expandedHash[EquihashType::HashWordsEx];
    ByteToUint32 v32;

    blake2b_state localState = *state;
    const uint32_t hashInputIdxLE = htole32_device(hashInputIdx);
    blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&hashInputIdxLE), sizeof(hashInputIdxLE));
    blake2b_final_device(&localState, hash, BLAKE2B_OUTBYTES);

    uint32_t curHashIdx = __umul24(hashInputIdx, EquihashType::IndicesPerHashOutput);
    uint32_t hashByteOffset = 0;
    for (uint32_t j = 0; j < EquihashType::IndicesPerHashOutput; ++j)
    {
        // map the output hash index to the appropriate bucket
        // and store the hash in the corresponding bucket
        // index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
        //   B = bucket index, N = hash index
		// bucketHashIndices for the first round stores the hash index directly

		// get first 11 bits of the hash for the bucket index
		v32.u4.x = hash[hashByteOffset];
		v32.u4.y = hash[hashByteOffset + 1];
		v32.u4.z = bswap_8(hash[hashByteOffset + 2]);
        v32.u4.w = 0;
		const uint32_t bucketIdx = v32.value & EquihashType::NBucketIdxMask;
            
		// try to increase number of hashes in the selected bucket
        const uint32_t hashIdxInBucket = atomicAdd(&bucketHashCounters[bucketIdx], 1);
        if (hashIdxInBucket >= EquihashType::NBucketSizeExtra)
        {
            atomicAdd(discardedCounter, 1);
            atomicSub(&bucketHashCounters[bucketIdx], 1);
            continue;
        }
        // find the place where to store the hash (some extra space exists in each bucket)
        const uint32_t bucketStorageHashIdx = __umul24(bucketIdx, EquihashType::NBucketSizeExtra) + hashIdxInBucket;
        const uint32_t bucketStorageHashIdxPtr = bucketStorageHashIdx * EquihashType::HashWordsEx;

        bucketHashIndices[bucketStorageHashIdx] = curHashIdx;

		// copy and expand hash for each collision round (bucket index + rest bits)
		expandedHash[0] = ((v32.value >> EquihashType::NBucketIdxBits) & EquihashType::NRestBitsMask) << 16 |
            bucketIdx;

        uint32_t bitOffset = EquihashType::CollisionBitLength;
        for (uint32_t k = 1; k < EquihashType::WK + 1; ++k, bitOffset += EquihashType::CollisionBitLength)
        {
			const uint32_t byteOffset = bitOffset >> 3; // bitOffset / 8
			const uint32_t bitOffsetInWord = bitOffset & 7; // bitOffset % 8
            const bool bNeedSwap = bitOffsetInWord == 0;

            const uint32_t offset = hashByteOffset + byteOffset;
			v32.u4.x = bNeedSwap ? hash[offset] : bswap_8(hash[offset]);
			v32.u4.y = hash[offset + 1];
			v32.u4.z = !bNeedSwap ? hash[offset + 2] : bswap_8(hash[offset + 2]);
			v32.u4.w = 0;
			const uint32_t hashDigit = (v32.value >> bitOffsetInWord) & EquihashType::CollisionBitMask;

            expandedHash[k] =
                ((hashDigit >> EquihashType::NBucketIdxBits) << 16) | (hashDigit & EquihashType::NBucketIdxMask);
        }
        memcpy(hashes + bucketStorageHashIdxPtr, expandedHash, sizeof(expandedHash));

        hashByteOffset += EquihashType::SingleHashOutput;
        ++curHashIdx;
    }
}

template<typename EquihashType>
void EhDevice<EquihashType>::generateInitialHashes()
{
	int minGridSize = 0;
	int blockSize = 0;
    if (bUseNewSolver)
	    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
            cudaKernel_generateInitialHashes_new<EquihashType>, 0, EquihashType::Base);
	else
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			cudaKernel_generateInitialHashes<EquihashType>, 0, EquihashType::Base);
	int gridSize = (EquihashType::Base + blockSize - 1) / blockSize;
    if (gridSize > CudaMaxGridSize.x)
    {
		gridSize = CudaMaxGridSize.x;
		blockSize = (EquihashType::Base + gridSize - 1) / gridSize;
        // Round up to the nearest multiple of 32 (warp size)
		blockSize = ((blockSize + 31) / 32) * 32;
    }

    dim3 gridDim(gridSize);
    dim3 blockDim(blockSize);
    
    if (bUseNewSolver)
        cudaKernel_generateInitialHashes_new<EquihashType><<<gridDim, blockDim>>>(
            d_initialState.get(),
            d_hashes.get(),
            d_bucketHashIndices.get(),
            d_bucketHashCounters.get(),
            d_discardedCounter.get());
    else
        cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(
            d_initialState.get(),
            d_hashes.get(), 
            d_bucketHashIndices.get(),
            d_bucketHashCounters.get(),
            d_discardedCounter.get());

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
        d_bucketHashCounters.get() + round * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount);
    copyToHost(vBucketHashIndices.data(), d_bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
        EquihashType::NHashStorageCount * sizeof(uint32_t));

    uint32_t nMaxCount = 0;
    m_dbgFile << "------\n";
    m_dbgFile << endl << "Bucket hash indices for round #" << round + 1 << ":" << endl;
    for (size_t i = 0; i < EquihashType::NBucketCount; ++i)
    {
        if (vBucketHashCounters[i] == 0)
            continue;
        if (vBucketHashCounters[i] > nMaxCount)
            nMaxCount = vBucketHashCounters[i];
        m_dbgFile << strprintf("\nRound %u, Bucket #%u, %u hash indices: ", round + 1, i, vBucketHashCounters[i]);
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
    m_dbgFile << "\nRound " << dec << round + 1 << " max hash indices per bucket: " << dec << nMaxCount << endl;
}

template <typename EquihashType>
void EhDevice<EquihashType>::debugPrintBucketCounters(const uint32_t bucketIdx, const uint32_t *collisionCountersPtr)
{
    v_uint32 vBucketCollisionCounts(EquihashType::NBucketCount);
    copyToHost(vBucketCollisionCounts.data(), collisionCountersPtr, vBucketCollisionCounts.size() * sizeof(uint32_t));

	string sInfo = strprintf("Round #%u [buckets %u/%u] hashes: ", round + 1, bucketIdx + 1, EquihashType::NBucketCount);
	for (uint32_t i = 0; i < EquihashType::NBucketCount; ++i)
		sInfo += to_string(vBucketCollisionCounts[i]) + " ";
#ifdef _WIN32_
	debug(sInfo);
#else
	printf(sInfo.c_str());
#endif
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

template <typename EquihashType>
__global__ void cudaKernel_processCollisions(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* bucketHashIndicesPrevPtr,  // bucketHashIndices  + __umul24(round,     EquihashType::NHashStorageCount)
    uint32_t* bucketHashIndicesPtr,      // bucketHashIndices  + __umul24(round + 1, EquihashType::NHashStorageCount)
    uint32_t* bucketHashCountersPrevPtr, // bucketHashCounters + __umul24(round,     EquihashType::NBucketCount)
    uint32_t* bucketHashCountersPtr,     // bucketHashCounters + __umul24(round + 1, EquihashType::NBucketCount);
    uint32_t* collisionPairs, 
    const uint32_t* collisionOffsets,
    uint32_t* collisionCounters,
    uint32_t* discardedCounter,
    const uint32_t round,
    const uint32_t maxCollisionsPerBucket,
    const uint32_t wordOffset, const uint64_t collisionBitMask,
    const uint32_t xoredBitOffset, const uint32_t xoredWordOffset)
{
	const uint32_t bucketIdx = blockIdx.x;
    if (bucketIdx >= EquihashType::NBucketCount)
        return;

	const uint32_t tid = threadIdx.x;
    const uint32_t threadsPerBlock = blockDim.x;
    const auto collisionPairsPtr = collisionPairs + __umul24(bucketIdx, maxCollisionsPerBucket);
    const auto collisionBucketOffset = collisionOffsets[bucketIdx];
    const uint32_t startIdxStorage = __umul24(bucketIdx, EquihashType::NBucketSizeExtra);
    const uint32_t hashCount = bucketHashCountersPrevPtr[bucketIdx];
    uint32_t xoredHash[EquihashType::HashWords];
    if (hashCount == 0)
        return;

    const bool bLastRound = round == EquihashType::WK - 1;

    // Allocate shared memory for hash indices in the current bucket used for this collision detection round
    __shared__ uint64_t sharedMaskedIndices[EquihashType::NBucketSizeExtra];

    // Load masked hashes into shared memory
    for (uint32_t i = tid; i < hashCount; i += threadsPerBlock)
    {
		const uint32_t hashWordIdx = (startIdxStorage + i) * EquihashType::HashWords + wordOffset;
		sharedMaskedIndices[i] =
			((static_cast<uint64_t>(hashes[hashWordIdx + 1]) << 32) | hashes[hashWordIdx]) & collisionBitMask;
    }
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    for (uint32_t leftPairIdx = tid; leftPairIdx < hashCount - 1; leftPairIdx += threadsPerBlock)
    {
		const uint32_t maskedIndexLeft = sharedMaskedIndices[leftPairIdx];

        for (uint32_t rightPairIdx = leftPairIdx + 1;  rightPairIdx < hashCount; ++rightPairIdx)
        {
			if (maskedIndexLeft != sharedMaskedIndices[rightPairIdx])
				continue;

            // hash collision found - xor the hashes and store the result
            const uint32_t rightStorageIdx = startIdxStorage + rightPairIdx;
            const uint32_t leftStorageIdx = startIdxStorage + leftPairIdx;

            const uint32_t hashIdxRight = rightStorageIdx * EquihashType::HashWords;
            const uint32_t hashIdxLeft = leftStorageIdx * EquihashType::HashWords;
            bool bAllZeroes = true;

            for (uint32_t j = wordOffset; j < EquihashType::HashWords; ++j)
            {
                const uint32_t xoredWord = hashes[hashIdxLeft + j] ^ hashes[hashIdxRight + j];
				bAllZeroes &= (xoredWord == 0);
                xoredHash[j] = xoredWord;
            }
            // accept all zeroes hash result at the last round
            if (bAllZeroes && !bLastRound)
                continue; // skip if all zeroes

            if (round > 0)
            {
                // skip this collision if it is based on the hash pair from the same bucket and with repeated previous collision indices
                const auto prevHashIdx1 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[leftStorageIdx]);
                const auto prevHashIdx2 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[rightStorageIdx]);
                if ((prevHashIdx1.x == prevHashIdx2.x) && 
                    !haveDistinctCollisionIndices(prevHashIdx1.y, prevHashIdx2.y, 
                        collisionPairs + __umul24(prevHashIdx1.x, maxCollisionsPerBucket)))
                    continue;
            }

            // define xored hash bucket based on the first NBucketIdxMask bits (starting from the CollisionBitLength)                
            uint32_t xoredBucketIdx = 
                static_cast<uint32_t>(((static_cast<uint64_t>(xoredHash[xoredWordOffset + 1]) << 32) | 
                                                              xoredHash[xoredWordOffset]) >> xoredBitOffset);
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
            uint32_t* ptr = xoredHashes + xoredBucketHashIdxStoragePtr;

            for (uint32_t j = 0; j < wordOffset; ++j)
                ptr[j] = 0;

			for (uint32_t j = wordOffset; j < EquihashType::HashWords; ++j)
				ptr[j] = xoredHash[j];

            const uint32_t collisionsInBucket = atomicAdd(&collisionCounters[bucketIdx], 1);
            const uint32_t collisionPairIdx = collisionBucketOffset + collisionsInBucket;

            // hash index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
            // B = bucket index, N = collision pair index
            bucketHashIndicesPtr[xoredBucketHashIdxStorage] = (bucketIdx << 16) | collisionPairIdx;
            collisionPairsPtr[collisionPairIdx] = (rightPairIdx << 16) | leftPairIdx;
        }
    }
}

template <typename EquihashType>
__global__ void cudaKernel_processCollisions_new(
    const uint32_t* hashes, uint32_t* xoredHashes,
    uint32_t* bucketHashIndicesPrevPtr,  // bucketHashIndices  + __umul24(round,     EquihashType::NHashStorageCount)
    uint32_t* bucketHashIndicesPtr,      // bucketHashIndices  + __umul24(round + 1, EquihashType::NHashStorageCount)
    uint32_t* bucketHashCountersPrevPtr, // bucketHashCounters + __umul24(round,     EquihashType::NBucketCount)
    uint32_t* bucketHashCountersPtr,     // bucketHashCounters + __umul24(round + 1, EquihashType::NBucketCount);
    uint32_t* collisionPairs, 
    const uint32_t* collisionOffsets,
    uint32_t* collisionCounters,
    uint32_t* discardedCounter,
    const uint32_t round,
    const uint32_t maxCollisionsPerBucket)
{
	const uint32_t tid = threadIdx.x;
    const uint32_t threadsPerBlock = blockDim.x;
    const uint32_t bucketIdx = blockIdx.x;
    __shared__ uint32_t hashCount;
    __shared__ uint32_t* collisionPairsPtr;
    __shared__ uint32_t startIdxStorage;
    __shared__ bool bLastRound;
    __shared__ uint32_t collisionBucketOffset;
    if (tid == 0)
    {
        if (bucketIdx >= EquihashType::NBucketCount)
            hashCount = 0;
        else
        {
            hashCount = bucketHashCountersPrevPtr[bucketIdx];
            collisionPairsPtr = collisionPairs + __umul24(bucketIdx, maxCollisionsPerBucket);
            startIdxStorage = __umul24(bucketIdx, EquihashType::NBucketSizeExtra);
            collisionBucketOffset = collisionOffsets[bucketIdx];
            bLastRound = round == EquihashType::WK - 1;
        }
    }
    __syncthreads();
    if (hashCount == 0)
        return;

	// Allocate shared memory for hash indices in the current bucket used for this collision detection round
    __shared__ uint16_t sharedIndices[EquihashType::NBucketSizeExtra];
    for (uint32_t i = tid; i < hashCount; i += threadsPerBlock)
        sharedIndices[i] = static_cast<uint16_t>(hashes[(startIdxStorage + i) * EquihashType::HashWordsEx + round] >> 16);
    __syncthreads(); // Ensure all threads have loaded data into shared memory

    uint32_t xoredHash[EquihashType::HashWordsEx];
    for (uint32_t leftPairIdx = tid; leftPairIdx < hashCount - 1; leftPairIdx += threadsPerBlock)
    {
		const uint32_t indexLeft = sharedIndices[leftPairIdx];
        for (uint32_t rightPairIdx = leftPairIdx + 1;  rightPairIdx < hashCount; ++rightPairIdx)
        {
			if (indexLeft != sharedIndices[rightPairIdx])
				continue;

			// hash collision found - xor the hashes and store the result
			const uint32_t rightStorageIdx = startIdxStorage + rightPairIdx;
			const uint32_t hashIdxRight = rightStorageIdx * EquihashType::HashWordsEx;

    		const uint32_t leftStorageIdx = startIdxStorage + leftPairIdx;
            const uint32_t hashIdxLeft = leftStorageIdx * EquihashType::HashWordsEx;

            bool bAllZeroes = true;
            for (uint32_t j = round + 1; j < EquihashType::HashWordsEx; ++j)
            {
				const uint32_t xoredWord = hashes[hashIdxLeft + j] ^ hashes[hashIdxRight + j];
				bAllZeroes &= (xoredWord == 0);
                xoredHash[j] = xoredWord;
            }

            // accept all zeroes hash result at the last round
            if (bAllZeroes && !bLastRound)
                continue; // skip if all zeroes

            if (round > 0)
            {
                // skip this collision if it is based on the hash pair from the same bucket and with repeated previous collision indices
                const auto prevHashIdx1 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[leftStorageIdx]);
                const auto prevHashIdx2 = getHashIndex<EquihashType>(bucketHashIndicesPrevPtr[rightStorageIdx]);
                if ((prevHashIdx1.x == prevHashIdx2.x) && 
                    !haveDistinctCollisionIndices(prevHashIdx1.y, prevHashIdx2.y, 
                        collisionPairs + __umul24(prevHashIdx1.x, maxCollisionsPerBucket)))
                    continue;
            }

            // define xored hash bucket index
            uint32_t xoredBucketIdx = static_cast<uint16_t>(xoredHash[round + 1]);
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
            const uint32_t xoredBucketHashIdxStorage = __umul24(xoredBucketIdx, EquihashType::NBucketSizeExtra) + xoredHashIdxInBucket;
            const uint32_t xoredBucketHashIdxStoragePtr = xoredBucketHashIdxStorage * EquihashType::HashWordsEx;

            uint32_t* ptr = xoredHashes + xoredBucketHashIdxStoragePtr;
            ptr[round] = 0;
			for (uint32_t j = round + 1; j < EquihashType::HashWordsEx; ++j)
				ptr[j] = xoredHash[j];

            const uint32_t collisionsInBucket = atomicAdd(&collisionCounters[bucketIdx], 1);
            const uint32_t collisionPairIdx = collisionBucketOffset + collisionsInBucket;

            // hash index format: [BBBB BBBB BBBB BBBB] [NNNN NNNN NNNN NNNN]
            // B = bucket index, N = collision pair index
            bucketHashIndicesPtr[xoredBucketHashIdxStorage] = (bucketIdx << 16) | collisionPairIdx;
            collisionPairsPtr[collisionPairIdx] = (rightPairIdx << 16) | leftPairIdx;
        }
    }
}

template <typename EquihashType>
void EhDevice<EquihashType>::processCollisions()
{
	int minGridSize = 0;
    int blockSize = 0; ; // desired number of threads per block

	if (bUseNewSolver)
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			cudaKernel_processCollisions_new<EquihashType>, sizeof(uint16_t) * EquihashType::NBucketSizeExtra, EquihashType::NBucketSizeExtra);
	else
		cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
			cudaKernel_processCollisions<EquihashType>, sizeof(uint64_t) * EquihashType::NBucketSizeExtra, EquihashType::NBucketSizeExtra);

    if (blockSize > EquihashType::NBucketSizeExtra)
	{
		blockSize = EquihashType::NBucketSizeExtra;
		blockSize = ((blockSize + 31) / 32) * 32;
	}

	dim3 gridDim(EquihashType::NBucketCount);
	dim3 blockDim(blockSize);

    try {
        const uint32_t nHashStorageWords = bUseNewSolver ? EquihashType::NHashStorageWordsEx : EquihashType::NHashStorageWords;

        cudaMemset(d_discardedCounter.get(), 0, sizeof(uint32_t));
        cudaMemset(d_collisionCounters.get(), 0, EquihashType::NBucketCount * sizeof(uint32_t));
        cudaMemset(d_xoredHashes.get(), 0, nHashStorageWords * sizeof(uint32_t));

        if (bUseNewSolver)
            cudaKernel_processCollisions_new<EquihashType><<<gridDim, blockDim>>>(
                d_hashes.get(), d_xoredHashes.get(),
                d_bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
                d_bucketHashIndices.get() + (round + 1) * EquihashType::NHashStorageCount,
                d_bucketHashCounters.get() + round * EquihashType::NBucketCount,
                d_bucketHashCounters.get() + (round + 1) * EquihashType::NBucketCount,
                d_collisionPairs.get(),
                d_collisionOffsets.get() + round * EquihashType::NBucketCount, 
                d_collisionCounters.get(),
                d_discardedCounter.get(),
                round,
                MaxCollisionsPerBucket);
        else
            cudaKernel_processCollisions<EquihashType><<<gridDim, blockDim>>>(
                d_hashes.get(), d_xoredHashes.get(),
                d_bucketHashIndices.get() + round * EquihashType::NHashStorageCount,
                d_bucketHashIndices.get() + (round + 1) * EquihashType::NHashStorageCount,
                d_bucketHashCounters.get() + round * EquihashType::NBucketCount,
                d_bucketHashCounters.get() + (round + 1) * EquihashType::NBucketCount,
                d_collisionPairs.get(),
                d_collisionOffsets.get() + round * EquihashType::NBucketCount, 
                d_collisionCounters.get(),
                d_discardedCounter.get(),
                round,
                MaxCollisionsPerBucket,
                EquihashType::HashWordOffsets[round],
                EquihashType::HashCollisionMasks[round],
                EquihashType::XoredHashBitOffsets[round],
                EquihashType::XoredHashWordOffsets[round]);

        // Check for any CUDA errors
        cudaError_t cudaError = cudaGetLastError();
        if (cudaError != cudaSuccess)
        {
            cerr << "CUDA error: " << cudaGetErrorString(cudaError) << endl;
            throw runtime_error("CUDA kernel launch failed");
        }
        cudaDeviceSynchronize();

        // Copy the collision counters from device to host
        copyToHost(vCollisionCounters.data(), d_collisionCounters.get(), EquihashType::NBucketCountStorageSize);

        // Store the accumulated collision pair offset for the current round
        for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
            vCollisionPairsOffsets[bucketIdx] += vCollisionCounters[bucketIdx];
        
        copyToDevice(d_collisionOffsets.get() + (round + 1) * EquihashType::NBucketCount, vCollisionPairsOffsets.data(),
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

    for (uint32_t mainIndex = 0; mainIndex < hashCount; ++mainIndex, hashIdx += EquihashType::HashWords)
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
        solutions[nSolutionCount].mainIndex = mainIndex;

        if (++nSolutionCount >= maxSolutionCount)
            break;
    }
    *solutionCount = nSolutionCount;
}

template<typename EquihashType>
__global__ void cudaKernel_findSolutions_new(
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

    for (uint32_t mainIndex = 0; mainIndex < hashCount; 
        ++mainIndex, hashIdx += EquihashType::HashWordsEx)
    {
        auto hashPtr = hashes + hashIdx + EquihashType::HashWordsEx - 2;
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
        solutions[nSolutionCount].mainIndex = mainIndex;

        if (++nSolutionCount >= maxSolutionCount)
            break;
    }
    *solutionCount = nSolutionCount;
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::findSolutions()
{
    uint32_t numSolutions = 0;
    cudaMemset(d_solutionCount.get(), 0, sizeof(uint32_t));

    if (bUseNewSolver)
        cudaKernel_findSolutions_new<EquihashType><<<1, 1>>>(
            d_hashes.get(),
            d_bucketHashCounters.get(),
            d_bucketHashIndices.get(),
            d_collisionPairs.get(),
            d_solutions.get(), d_solutionCount.get(),
            MaxCollisionsPerBucket,
            MaxSolutions);
    else
        cudaKernel_findSolutions<EquihashType><<<1, 1>>>(
            d_hashes.get(),
            d_bucketHashCounters.get(),
            d_bucketHashIndices.get(),
            d_collisionPairs.get(),
            d_solutions.get(), d_solutionCount.get(),
            MaxCollisionsPerBucket,
            MaxSolutions);

    cudaDeviceSynchronize();

    copyToHost(&numSolutions, d_solutionCount.get(), sizeof(uint32_t));
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
    copyToHost(vBucketHashCounters.data(), d_bucketHashCounters.get() + (bInitialHashes ? 0 : (round + 1) * EquihashType::NBucketCount),
        EquihashType::NBucketCount * sizeof(uint32_t));

    v_uint32 vBucketHashIndices(EquihashType::NHashStorageCount, 0);
    copyToHost(vBucketHashIndices.data(), d_bucketHashIndices.get() + (bInitialHashes ? 0 : (round + 1) * EquihashType::NHashStorageCount),
        vBucketHashIndices.size() * sizeof(uint32_t));
    
	const uint32_t nHashStorageWords = bUseNewSolver ? EquihashType::NHashStorageWordsEx : EquihashType::NHashStorageWords;
	const uint32_t nHashWords = bUseNewSolver ? EquihashType::HashWordsEx : EquihashType::HashWords;

    v_uint32 vHostHashes;
    vHostHashes.resize(nHashStorageWords);
    copyToHost(vHostHashes.data(), d_hashes.get(), nHashStorageWords * sizeof(uint32_t));

    uint32_t nDiscarded = 0;
    copyToHost(&nDiscarded, d_discardedCounter.get(), sizeof(uint32_t));

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
            for (size_t j = 0; j < nHashWords; ++j)
            {
                const uint32_t hashInputIdx = (bucketHashStorageIdx + i) * nHashWords + j;
                if (vHostHashes[hashInputIdx])
                    bAllZeroes = false;
				if (bUseNewSolver)
                    sLog += strprintf("%08x ", vHostHashes[hashInputIdx]);
                else
                    sLog += strprintf("%08x ", htonl(vHostHashes[hashInputIdx]));
            }
            sLog += strprintf("| %u ", vBucketHashIndices[bucketHashStorageIdx + i]);
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
    copyToHost(vBucketHashCounters.data(), d_bucketHashCounters.get() + 
        (round + (bInitialHashes ? 0 : 1)) * EquihashType::NBucketCount,
        EquihashType::NBucketCount * sizeof(uint32_t));

    uint32_t nDiscarded = 0;
    copyToHost(&nDiscarded, d_discardedCounter.get(), sizeof(uint32_t));
#ifdef _WIN32_
	debug("Discarded: %u", nDiscarded);
#else
	printf("Discarded: %u\n", nDiscarded);
#endif
    const uint32_t nHashStorageWords = bUseNewSolver ? EquihashType::NHashStorageWordsEx : EquihashType::NHashStorageWords;
	const uint32_t nHashWords = bUseNewSolver ? EquihashType::HashWordsEx : EquihashType::HashWords;

    v_uint32 vHostHashes;
    vHostHashes.resize(nHashStorageWords);
    copyToHost(vHostHashes.data(), d_hashes.get(), nHashStorageWords * sizeof(uint32_t));
    
    v_uint32 hostHashes;
    string sLog;
    sLog.reserve(1024);
	if (bInitialHashes)
		sLog = strprintf("\nInitial hashes (discarded - %u):\n", nDiscarded);
	else
		sLog = strprintf("\nHashes for round #%u (discarded - %u):\n", round + 1, nDiscarded);
#ifdef _WIN32_
	gl_console_logger->info(sLog);
#else
	printf(sLog.c_str());
#endif

    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        if ((bucketIdx > 3) && (bucketIdx < EquihashType::NBucketCount - 3))
            continue;
        size_t nBucketHashCount = vBucketHashCounters[bucketIdx];
        if (nBucketHashCount == 0)
            continue;
#ifdef _WIN32_
        gl_console_logger->info("");
        if (bInitialHashes)
			gl_console_logger->info("Initial bucket #{}, ({}) hashes:", bucketIdx, nBucketHashCount);
        else
			gl_console_logger->info("Round %u bucket #{}, ({}) hashes:", round + 1, bucketIdx, nBucketHashCount);
#else
	printf("\n");
        if (bInitialHashes)
			printf("Initial bucket #%u, (%zu) hashes:", bucketIdx, nBucketHashCount);
        else
			printf("Round %u bucket #%u, (%zu) hashes:", round + 1, bucketIdx, nBucketHashCount);
#endif

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
            for (size_t j = 0; j < nHashWords; ++j)
            {
                const uint32_t hashInputIdx = (bucketStorageIdx + i) * nHashWords + j;
                if (vHostHashes[hashInputIdx])
                    bAllZeroes = false;
                if (bUseNewSolver)
                    sLog += strprintf("%08x ", vHostHashes[hashInputIdx]);
                else
                    sLog += strprintf("%08x ", htonl(vHostHashes[hashInputIdx]));
            }
            if (bAllZeroes)
                sLog += " (all zeroes)";
#ifdef _WIN32_
			gl_console_logger->info(sLog);
#else
			printf(sLog.c_str());
#endif

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
        d_collisionCounters.get(), EquihashType::NBucketCount * sizeof(uint32_t));

    copyToHost(vCollPairsOffsets.data(),
        d_collisionOffsets.get() + round * EquihashType::NBucketCount,
        vCollPairsOffsets.size() * sizeof(uint32_t));

    v_uint32 vCollisionPairs(EquihashType::NBucketCount * MaxCollisionsPerBucket);
    copyToHost(vCollisionPairs.data(), 
        d_collisionPairs.get(), vCollisionPairs.size() * sizeof(uint32_t));

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
                    m_dbgFile << endl;
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
        d_collisionCounters.get(),
        vBucketCollisionCounts.size() * sizeof(uint32_t));
    if (round > 0)
        copyToHost(vCollisionPairsOffsets.data(),
            d_collisionOffsets.get() + round * EquihashType::NBucketCount,
            vCollisionPairsOffsets.size() * sizeof(uint32_t));

    constexpr uint32_t COLLISIONS_PER_LINE = 10;
#ifdef _WIN32_
	debug("Collision pairs for round #{}:", round + 1);
#else
	printf("Collision pairs for round #%d:\n", round + 1);
#endif

    string sLog;
    for (uint32_t bucketIdx = 0; bucketIdx < EquihashType::NBucketCount; ++bucketIdx)
    {
        if ((bucketIdx > 3) && (bucketIdx < EquihashType::NBucketCount - 3))
            continue;
        size_t nBucketCollisionCount = vBucketCollisionCounts[bucketIdx];
        if (nBucketCollisionCount == 0)
            continue;
#ifdef _WIN32_
		debug("Round %u, Bucket #%u, %u collision pairs:", round + 1, bucketIdx, nBucketCollisionCount);
#else
		printf("Round %u, Bucket #%u, %u collision pairs:\n", round + 1, bucketIdx, nBucketCollisionCount);
#endif

        v_uint32 hostCollisionPairs(nBucketCollisionCount);
        copyToHost(hostCollisionPairs.data(), 
            d_collisionPairs.get() + bucketIdx * MaxCollisionsPerBucket + vCollisionPairsOffsets[bucketIdx], 
            nBucketCollisionCount * sizeof(uint32_t));

        uint32_t nPairNo = 0;
        sLog.clear();
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
                {
#ifdef _WIN32_
                    debug(sLog);
#else
		printf(sLog.c_str());
#endif
					sLog.clear();
                }
				sLog = strprintf("\nPairInfo %u:", i);
            }
			sLog += strprintf(" (%u-%u)", idxLeft, idxRight);
        }
        if (!sLog.empty())
        {
#ifdef _WIN32_
            debug(sLog);
#else
		printf(sLog.c_str());
#endif
            sLog.clear();
        }
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
#ifdef _WIN32_
	debug("Max collision pair offset: %u, max collision pair count: %u", maxCollisionPairOffset, maxCollisionPairCount);
#else
	printf("Max collision pair offset: %u, max collision pair count: %u\n", maxCollisionPairOffset, maxCollisionPairCount);
#endif
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
    copyToHost(&nSolutionCount, d_solutionCount.get(), sizeof(uint32_t));

    // Resize the host solutions vector
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), d_solutions.get(), nSolutionCount * sizeof(typename EquihashType::solution_type));
}

template<typename EquihashType>
uint32_t EhDevice<EquihashType>::solve()
{
    clear();

    EQUI_TIMER_DEFINE_EX(total);
	EQUI_TIMER_START_EX(total);

    // Generate initial hash values
    EQUI_TIMER_DEFINE;
    EQUI_TIMER_START;
    generateInitialHashes();
	if (bBreakSolver)
		return 0;

    EQUI_TIMER_STOP("Initial hash generation");
    DEBUG_FN(debugPrintHashes(true));
    DBG_EQUI_WRITE_FN(debugWriteHashes(true));
    DBG_EQUI_WRITE_FN(debugWriteBucketIndices());

    // Perform K rounds of collision detection and XORing
    while (!bBreakSolver && (round < EquihashType::WK))
    {
        // Detect collisions and XOR the colliding hashes
        EQUI_TIMER_START;
        processCollisions();
        EQUI_TIMER_STOP(strprintf("Round [%u], collisions", round + 1));

		if (bBreakSolver)
			break;

        DEBUG_FN(debugPrintCollisionPairs());
        DBG_EQUI_WRITE_FN(debugWriteCollisionPairs());
        // Swap the hash pointers for the next round
        swap(d_hashes, d_xoredHashes);
        DEBUG_FN(debugPrintHashes(false));
        DBG_EQUI_WRITE_FN(debugWriteHashes(false));

        ++round;

        //cout << "Round #" << dec << round << " completed" << endl;
        DBG_EQUI_WRITE_FN(debugWriteBucketIndices());
    }

	if (bBreakSolver)
		return 0;

    EQUI_TIMER_START;
    uint32_t nSolutionCount = findSolutions();
    EQUI_TIMER_STOP("Solution search");
	EQUI_TIMER_STOP_EX(total, "Total solver time");
    return nSolutionCount;
}

// Explicit template instantiation
template class EhDevice<Eh200_9>;
