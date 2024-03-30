// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <vector>

#include <blake2b.h>
#include <local_types.h>

template<typename EquihashType>
class EhDevice
{
public:
        EhDevice() noexcept = default;
        ~EhDevice();

        std::unique_ptr<blake2b_state> initialState;
        std::unique_ptr<uint32_t, CudaDeleter> hashes;
        std::unique_ptr<uint32_t, CudaDeleter> xoredHashes;

        std::vector<std::unique_ptr<uint32_t, CudaDeleter>> vCollisionPairs;
        std::vector<std::vector<std::unique_ptr<uint32_t, CudaDeleter>>> vCollisionCounters;

        std::unique_ptr<typename EquihashType::solution, CudaDeleter> solutions;
        std::unique_ptr<uint32_t, CudaDeleter> solutionCount;

        uint32_t round = 0;

        bool allocate_memory();

        static inline constexpr ThreadsPerBlock = 256;

        void generateInitialHashes();
        void detectCollisions();
        void xorCollisions();
        uint32_t findSolutions();

private:
    // Accumulated collision pair offsets for each round
    v_uint32 m_vCollisionPairsOffsets;
};

template<typename EquihashType>
void copySolutionsToHost(typename EquihashType::solution* devSolutions, const uint32_t nSolutionCount, 
    std::vector<typename EquihashType::solution>& vHostSolutions);
