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

    std::unique_ptr<blake2b_state, GpuMemDeleter> initialState;
    std::unique_ptr<uint32_t, GpuMemDeleter> hashes;
    std::unique_ptr<uint32_t, GpuMemDeleter> xoredHashes;

    std::unique_ptr<uint32_t, GpuMemDeleter> collisionPairs;
    // Accumulated collision pair offsets for each round
    v_uint32 vCollisionPairsOffsets;

    std::unique_ptr<uint32_t, GpuMemDeleter> collisionCounters;   

    std::unique_ptr<typename EquihashType::solution, GpuMemDeleter> solutions;
    std::unique_ptr<uint32_t, GpuMemDeleter> solutionCount;

    uint32_t round = 0;

    bool allocate_memory();
    uint32_t solver();

    void copySolutionsToHost(std::vector<typename EquihashType::solution>& vHostSolutions);

    static inline constexpr uint32_t ThreadsPerBlock = 256;
    static inline constexpr uint32_t MaxCollisionsPerBucket = 50'000;

private:
    void generateInitialHashes();
    void detectCollisions();
    void xorCollisions();
    uint32_t findSolutions();

    void debugPrintHashes();
};
