// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <src/equihash/equihash-check.h>
#include <src/equihash/equihash.h>
#include <src/equihash/blake2b_host.h>
#include <src/equihash/block.h>
#include <src/utils/arith_uint256.h>
#include <src/utils/streams.h>
#include <blake2b.h>
#include <tinyformat.h>

using namespace std;

template <typename EquihashType>
bool CheckEquihashSolution(const CBlockHeader &block)
{
    // Hash state
    blake2b_state state;
    EquihashType::InitialiseState(state, DEFAULT_EQUIHASH_PERS_STRING);

    // I = the block header minus nonce and solution.
    CEquihashInput I{block};
    // I||V
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss.reserve(I.GetReserveSize() + block.nNonce.SIZE);
    ss << I;
    ss << block.nNonce;

    // H(I||V||...
    blake2b_update_host(&state, (unsigned char*)&ss[0], ss.size());

    const bool isValid = EquihashType::IsValidSolution(state, block.nSolution);
    if (!isValid)
    {
        cerr << "CheckEquihashSolution - Invalid solution" << endl;
        return false;
    }
    return true;
}

/**
 * Check proof of work for the block.
 * 
 * \param hashBlock - block header hash
 * \param nBits - difficulty target
 * \param powLimit - maximum allowed proof of work
 * \return true if the proof of work is correct
 */

bool CheckProofOfWork(const uint256& hashBlock, unsigned int nBits, const uint256& powLimit)
{
    bool fNegative;
    bool fOverflow;
    arith_uint256 bnTarget;

    bnTarget.SetCompact(nBits, &fNegative, &fOverflow);
    
    // Check range
    if (fNegative || bnTarget == 0 || fOverflow || bnTarget > UintToArith256(powLimit))
    {
        string sLog = strprintf("CheckProofOfWork - fNegative = %s, fOverflow = %s, bnTarget = %s, powLimit = %s \n",
                fNegative? "true": "false", 
                fOverflow? "true": "false", 
                ArithToUint256(bnTarget).ToString(),
                powLimit.ToString()
                );
        cerr << sLog << endl;
        return false;
    }

    // Check proof of work matches claimed amount
    if (UintToArith256(hashBlock) > bnTarget)
    {
        string sLog = strprintf("CheckProofOfWork - hash = %s, bnTarget = %s\n", 
                hashBlock.ToString(),
                ArithToUint256(bnTarget).ToString());
        cerr << sLog << endl;
        return false;
    }

    return true;
}