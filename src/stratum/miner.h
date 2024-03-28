// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <string>
#include <functional>

#include <blake2b.h>
#include <src/utils/uint256.h>
#include <src/stratum/client.h>

using funcGenerateNonce_t = std::function<const uint256 (uint32_t nExtraNonce2)>;
using funcSubmitSolution_t = std::function<void(const uint32_t nExtraNonce2, const std::string& sTime, 
        const std::string& sNonce, const std::string &sHexSolution)>;

template<typename EquihashType>
uint32_t miningLoop(const blake2b_state& initialState, uint32_t &nExtraNonce2, const std::string &sTime,
                    const size_t nIterations, const uint32_t threadsPerBlock,
                    const funcGenerateNonce_t &genNonceFn, const funcSubmitSolution_t &submitSolutionFn);

void miner(CStratumClient &client);
