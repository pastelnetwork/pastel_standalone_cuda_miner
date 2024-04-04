// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <algorithm>

#include <blake2b.h>
#include <src/utils/uint256.h>
#include <src/equihash/equihash-types.h>

constexpr auto DEFAULT_EQUIHASH_PERS_STRING = "ZcashPoW";

template<unsigned int N, unsigned int K>
class EquihashSolver : public Equihash<N, K>
{
public:
    bool InitializeState(blake2b_state &state, const std::string &sPersString);

    bool IsValidSolution(std::string &error, const blake2b_state& base_state, const v_uint8 &soln);
};

// the maximum number of solutions that can be found by the miner
constexpr uint32_t MAXSOLUTIONS = 10;

using EhSolver200_9 = EquihashSolver<200, 9>;

