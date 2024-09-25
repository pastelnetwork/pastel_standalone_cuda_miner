// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <src/utils/uint256.h>
#include <src/utils/arith_uint256.h>
#include <src/equihash/block.h>

/** Check whether the Equihash solution in a block header is valid */
template <typename EquihashType>
bool CheckEquihashSolution(const CBlockHeader &block);

/** Check whether a block hash satisfies the proof-of-work requirement specified by nBits */
bool CheckProofOfWork(const uint256& hashBlock, unsigned int nBits, const uint256& powLimit);
// Check whether the miner solution is below the target
bool CheckMinerSolution(const uint256& hashBlock, const arith_uint256& target);
