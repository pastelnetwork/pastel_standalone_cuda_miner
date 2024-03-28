
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

// Constants for tree storage
constexpr uint32_t BITS_PER_WORD = 32;
#define BITMAP_SIZE(numNodes) ((numNodes + BITS_PER_WORD - 1) / BITS_PER_WORD)

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
        uint32_t leBlockIdx = htole32(blockIdx);

        blake2b_state localState = *state;
        blake2b_update_device(&localState, reinterpret_cast<uint8_t*>(&leBlockIdx), sizeof(leBlockIdx));

        uint8_t hash[EquihashType::HashOut];  
        blake2b_final_device(&localState, hash, EquihashType::HashOut);

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

    cudaKernel_generateInitialHashes<EquihashType><<<gridDim, blockDim>>>(devState, devHashes, EquihashType::IndicesPerHashOutput);
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
            for (uint32_t i = 0; i < EquihashType::K; ++i)
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
void copySolutionsToHost(typename EquihashType::solution* devSolutions, uint32_t nSolutionCount, v_strings& resultSolutions)
{
    // Resize the host solutions vector
    vector<typename EquihashType::solution> vHostSolutions;
    vHostSolutions.resize(nSolutionCount);

    // Copy the solutions from device to host
    copyToHost(vHostSolutions.data(), devSolutions, nSolutionCount * sizeof(EquihashType::solution));

    // Process the solutions and store them in the result solutions vector
    resultSolutions.clear();
    resultSolutions.reserve(nSolutionCount);

    for (const auto& solution : vHostSolutions)
    {
        // Construct the block header using the solution indices
        string sHexSolution = HexStr(solution.indices, solution.indices + EquihashType::ProofSize);
        resultSolutions.push_back(sHexSolution);
    }
}

/*

// CUDA kernel to XOR hashes and store index trees
__global__ void xorHashesKernel(uint32_t* hashes, uint32_t* xoredHashes, uint32_t* indexes, 
                                uint32_t numHashes, uint32_t threadsPerBlock) 
{
    __shared__ uint32_t sharedHashes[NSLOTS * HASHWORDS];
    __shared__ uint32_t sharedXoredHashes[NSLOTS * HASHWORDS];
    __shared__ uint32_t sharedIndexes[NSLOTS];

    uint32_t hashIdx = blockIdx.x * threadsPerBlock + threadIdx.x;
    if (hashIdx >= numHashes)
        return;

    // Load hash into shared memory
    uint32_t slotIdx = threadIdx.x;
    for (uint32_t i = 0; i < HASHWORDS; i++)  
        sharedHashes[slotIdx * HASHWORDS + i] = hashes[hashIdx * HASHWORDS + i];
    
    // XOR with other hashes in the block and store in shared memory
    for (uint32_t stride = 1; stride < threadsPerBlock; stride *= 2)
    {
        uint32_t otherSlot = slotIdx ^ stride;
        for (uint32_t i = 0; i < HASHWORDS; i++)
            sharedXoredHashes[slotIdx * HASHWORDS + i] ^= 
                sharedHashes[otherSlot * HASHWORDS + i];
        if (slotIdx < stride)  // Only half the threads need to update indexes
            sharedIndexes[slotIdx] = (sharedIndexes[slotIdx] << 1) | 
                                      (sharedIndexes[slotIdx + stride] & 1);
    }

    if (slotIdx == 0)
    {
        uint32_t row = blockIdx.x / NSLOTS;
        uint32_t slot = blockIdx.x % NSLOTS;
        // Write XORed hash to global memory
        for (uint32_t i = 0; i < HASHWORDS; i++)
            xoredHashes[row * HASHWORDS + i] = sharedXoredHashes[i];
        // Write tree index to global memory  
        indexes[row * NINDEXES + slot] = sharedIndexes[0];
    }
}

// CUDA kernel to compare xored hashes and identify solutions
__global__ void findSolutionsKernel(uint32_t* xoredHashes, uint32_t* indexes,
                                    uint32_t* solutions, uint32_t maxSolsPerThread)
{
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t totalHashes = NSLOTS * (1 << WK);

    for (uint32_t i = tid; i < totalHashes; i += blockDim.x * gridDim.x)
    {
        uint32_t solCount = 0;  
        for (uint32_t j = i + 1; j < totalHashes && solCount < maxSolsPerThread; j++)
        {
            bool match = true;
            for (uint32_t k = 0; k < HASHWORDS; k++)
                match &= (xoredHashes[i * HASHWORDS + k] == xoredHashes[j * HASHWORDS + k]);

            if (match)
            {
                uint32_t solIdx = atomicAdd(solutions, 2);
                if (solIdx + 2 <= MAXSOLUTIONS)
                {
                    solutions[solIdx] = indexes[i];
                    solutions[solIdx + 1] = indexes[j];
                }
                solCount++;
            }
        }
    }
}

// CUDA kernel to perform final round of solution identification
__global__ void finalSolutionKernel(uint32_t* xoredHashes, uint32_t* indexes, uint32_t* solutions, uint32_t numHashes, uint32_t maxSolutions)
{
    __shared__ uint32_t sharedSolutions[MAX_SOLUTIONS_PER_BLOCK * 2];
    __shared__ uint32_t sharedSolutionCount;

    if (threadIdx.x == 0)
        sharedSolutionCount = 0;
        
    __syncthreads();

//    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = gid; i < numHashes; i += stride)
    {
        for (uint32_t j = i + 1; j < numHashes; j++)
        {
            bool match = true;
            for (uint32_t k = 0; k < HASHWORDS; k++)
            {
                if (xoredHashes[i * HASHWORDS + k] != xoredHashes[j * HASHWORDS + k])
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                uint32_t solIdx = atomicAdd(&sharedSolutionCount, 2);
                if (solIdx + 2 <= MAX_SOLUTIONS_PER_BLOCK * 2)
                {
                    sharedSolutions[solIdx] = indexes[i];
                    sharedSolutions[solIdx + 1] = indexes[j];
                }
            }
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && sharedSolutionCount > 0)
    {
        uint32_t solIdx = atomicAdd(solutions, sharedSolutionCount);
        if (solIdx + sharedSolutionCount <= maxSolutions)
        {
            for (uint32_t i = 0; i < sharedSolutionCount; i++)
                solutions[solIdx + i] = sharedSolutions[i];
        }
    }
}

// Launch the kernel to XOR hashes and store index trees
void launchXorHashesKernel(uint32_t* devHashes, uint32_t* devXoredHashes, uint32_t* devIndexes,
                           uint32_t numHashes, uint32_t threadsPerBlock)
{
    dim3 gridDim((numHashes + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDim(threadsPerBlock);

    xorHashesKernel<<<gridDim, blockDim>>>(devHashes, devXoredHashes, devIndexes, numHashes, threadsPerBlock);
}

// Launch the kernel to find solutions
void launchFindSolutionsKernel(uint32_t* devXoredHashes, uint32_t* devIndexes, uint32_t* devSolutions,
                               uint32_t numHashes, uint32_t threadsPerBlock, uint32_t maxSolsPerThread)
{
    dim3 gridDim((numHashes + threadsPerBlock - 1) / threadsPerBlock);
    dim3 blockDim(threadsPerBlock);

    findSolutionsKernel<<<gridDim, blockDim>>>(devXoredHashes, devIndexes, devSolutions, maxSolsPerThread);
}

// CUDA kernel to propagate XOR values up the tree
__global__ void propagateXorKernel(uint32_t* bitmap, uint32_t* xoredHashes, uint32_t numNodes)
{
    __shared__ uint32_t sharedBitmap[BITMAP_SIZE(NSLOTS)];
    __shared__ uint32_t sharedXoredHashes[NSLOTS * HASHWORDS];

    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t blockSize = blockDim.x;

    // Load bitmap and XORed hashes into shared memory
    for (uint32_t i = tid; i < BITMAP_SIZE(NSLOTS); i += blockSize)
        sharedBitmap[i] = bitmap[bid * BITMAP_SIZE(NSLOTS) + i];

    for (uint32_t i = tid; i < NSLOTS * HASHWORDS; i += blockSize)
        sharedXoredHashes[i] = xoredHashes[bid * NSLOTS * HASHWORDS + i];
    __syncthreads();

    // Propagate XOR values up the tree
    for (uint32_t i = 1; i < numNodes; i++)
    {
        uint32_t parent = (i - 1) / 2;
        uint32_t parentWord = parent / BITS_PER_WORD;
        uint32_t parentBit = parent % BITS_PER_WORD;

        if ((sharedBitmap[parentWord] & (1 << parentBit)) != 0)
        {
            uint32_t leftChild = i;
            uint32_t rightChild = i + 1;

            for (uint32_t j = tid; j < HASHWORDS; j += blockSize)
            {
                sharedXoredHashes[parent * HASHWORDS + j] ^=
                    sharedXoredHashes[leftChild * HASHWORDS + j] ^
                    sharedXoredHashes[rightChild * HASHWORDS + j];
            }
        }
        __syncthreads();
    }

    // Write updated XORed hashes back to global memory
    for (uint32_t i = tid; i < NSLOTS * HASHWORDS; i += blockSize)
        xoredHashes[bid * NSLOTS * HASHWORDS + i] = sharedXoredHashes[i];
}

// Launch the kernel to propagate XOR values up the tree
void launchPropagateXorKernel(uint32_t* deviceBitmap, uint32_t* deviceXoredHashes, uint32_t numNodes, uint32_t numBlocks)
{
    dim3 blockDim(256);
    dim3 gridDim(numBlocks);
    propagateXorKernel<<<gridDim, blockDim>>>(deviceBitmap, deviceXoredHashes, numNodes);
}

// Launch the final solution kernel
void launchFinalSolutionKernel(uint32_t* deviceXoredHashes, uint32_t* deviceIndexes, uint32_t* deviceSolutions, uint32_t numHashes, uint32_t maxSolutions)
{
    dim3 blockDim(256);
    dim3 gridDim((numHashes + blockDim.x - 1) / blockDim.x);
    finalSolutionKernel<<<gridDim, blockDim>>>(deviceXoredHashes, deviceIndexes, deviceSolutions, numHashes, maxSolutions);
}
*/