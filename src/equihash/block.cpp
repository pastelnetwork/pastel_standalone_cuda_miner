// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <src/equihash/block.h>
#include <src/utils/streams.h>
#include <src/utils/serialize.h>
#include <src/utils/strencodings.h>
#include <src/utils/hash.h>

using namespace std;

// Function to construct the block header using the solution indices
string serializeBlockHeader(const CBlockHeader& block, 
    const string& sTime, const string& sNonce, const string& solution)
{
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss << block.nVersion;
    ss << block.hashPrevBlock;
    ss << block.hashMerkleRoot;
    ss << block.hashFinalSaplingRoot;
    ss << block.nTime;
    ss << block.nBits;
    if (block.nVersion >= CBlockHeader::VERSION_SIGNED_BLOCK)
    {
        ss << block.sPastelID;
        ss << block.prevMerkleRootSignature;
    }
    ss << sNonce;
    ss << solution;

    uint256 hash = Hash(ss.begin(), ss.end());
    string blockHeader = HexStr(hash.begin(), hash.end());

    return blockHeader;
}