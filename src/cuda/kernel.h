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
    std::unique_ptr<uint32_t, CudaDeleter> hashes;
    std::unique_ptr<uint32_t, CudaDeleter> xoredHashes;

    std::unique_ptr<uint32_t, CudaDeleter> collisionPairs;
    // Accumulated collision pair offsets for each bucket
    v_uint32 vCollisionPairsOffsets;
    v_uint32 vPrevCollisionPairsOffsets;
    std::unique_ptr<uint32_t, CudaDeleter> collisionPairOffsets;
    std::unique_ptr<uint32_t, CudaDeleter> collisionCounters;   


    std::unique_ptr<typename EquihashType::solution, CudaDeleter> solutions;
    std::unique_ptr<uint32_t, CudaDeleter> solutionCount;

    uint32_t round = 0;

    bool allocate_memory();
    uint32_t solver();

    void copySolutionsToHost(std::vector<typename EquihashType::solution>& vHostSolutions);

    static inline constexpr uint32_t ThreadsPerBlock = 256;
    static inline constexpr uint32_t MaxCollisionsPerBucket = 100'000;

private:
    void generateInitialHashes();
    void detectCollisions();
    void xorCollisions();
    uint32_t findSolutions();

    void debugPrintHashes();
};
