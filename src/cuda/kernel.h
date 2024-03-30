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

    std::unique_ptr<uint32_t*, CudaDeleter> collisionPairs;
    std::vector<std::unique_ptr<uint32_t, CudaDeleter>> vBucketCollisionPairs;
    // Accumulated collision pair offsets for each round
    v_uint32 vCollisionPairsOffsets;

     std::unique_ptr<uint32_t*, CudaDeleter> collisionCounters;   
    std::vector<std::vector<std::unique_ptr<uint32_t, CudaDeleter>>> vCollisionCounters;

    std::unique_ptr<typename EquihashType::solution, CudaDeleter> solutions;
    std::unique_ptr<uint32_t, CudaDeleter> solutionCount;

    uint32_t round = 0;

    bool allocate_memory();
    uint32_t solver();

    static inline constexpr uint32_t ThreadsPerBlock = 256;
    void copySolutionsToHost(std::vector<typename EquihashType::solution>& vHostSolutions);

private:
    void generateInitialHashes();
    void detectCollisions();
    void xorCollisions();
    uint32_t findSolutions();

    void debugPrintHashes();
};
