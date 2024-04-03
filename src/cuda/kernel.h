// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <vector>
#include <memory>

#include <blake2b.h>
#include <local_types.h>
#include <src/cuda/memutils.h>

template<typename EquihashType>
class EhDevice
{
public:
    EhDevice() noexcept = default;
    ~EhDevice() {}

    std::unique_ptr<blake2b_state, CudaDeleter> initialState;
    std::unique_ptr<uint32_t, CudaDeleter> hashes;               // NHashStorageWords
    std::unique_ptr<uint32_t, CudaDeleter> xoredHashes;          // NHashStorageWords
    std::unique_ptr<uint32_t, CudaDeleter> bucketHashIndices;    // NHashStorageCount * (WK + 1)

    std::unique_ptr<uint32_t, CudaDeleter> collisionPairs;       // NBucketCount * MaxCollisionsPerBucket
    // Accumulated collision pair offsets for each bucket
    v_uint32 vCollisionPairsOffsets;                             // NBucketCount
    v_uint32 vPrevCollisionPairsOffsets;                         // NBucketCount
    std::unique_ptr<uint32_t, CudaDeleter> collisionPairOffsets; // NBucketCount
    std::unique_ptr<uint32_t, CudaDeleter> collisionCounters;    // NBucketCount * (WK + 1)
    v_uint32 vCollisionCounters;                                 // NBucketCount

    std::unique_ptr<typename EquihashType::solution, CudaDeleter> solutions; // MAXSOLUTIONS
    std::unique_ptr<uint32_t, CudaDeleter> solutionCount;        // 1

    uint32_t round = 0;

    bool allocate_memory();
    uint32_t solver();

    void copySolutionsToHost(std::vector<typename EquihashType::solution>& vHostSolutions);

    static inline constexpr uint32_t ThreadsPerBlock = 256;
    static inline constexpr uint32_t MaxCollisionsPerBucket = (EquihashType::WK + 1) * EquihashType::NBucketSize; // 10 * 65535 = 655350

private:
    void rebucketHashes();
    void generateInitialHashes();
    void processCollisions();
    uint32_t findSolutions();

    void debugPrintHashes(const bool bIsBucketed);
    void debugPrintXoredHashes();
    void debugPrintCollisionPairs();
    void debugPrintBucketCounters(const uint32_t bucketIdx, const uint32_t *collisionCountersPtr);
};
