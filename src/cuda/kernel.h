// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <vector>

#include <blake2b.h>
#include <local_types.h>

template <typename EquihashType>
void generateInitialHashes(const blake2b_state* devState, uint32_t* devHashes, const uint32_t threadsPerBlock);

template <typename EquihashType>
void detectCollisions(uint32_t* devHashes, uint32_t* devSlotBitmaps, const uint32_t threadsPerBlock);

template<typename EquihashType>
void xorCollisions(uint32_t* devHashes, uint32_t* devSlotBitmaps, uint32_t* devXoredHashes, const uint32_t threadsPerBlock);

template<typename EquihashType>
uint32_t findSolutions(uint32_t* devHashes, uint32_t* devSlotBitmaps, typename EquihashType::solution* devSolutions,
    uint32_t* devSolutionCount, const uint32_t threadsPerBlock);

template<typename EquihashType>
void copySolutionsToHost(typename EquihashType::solution* devSolutions, const uint32_t nSolutionCount, 
    std::vector<typename EquihashType::solution>& vHostSolutions);
