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

CBlockHeader::CBlockHeader() noexcept
{
    Clear();
}

CBlockHeader::CBlockHeader(CBlockHeader&& hdr) noexcept
{
    move_from(move(hdr));
}

CBlockHeader& CBlockHeader::operator=(CBlockHeader&& hdr) noexcept
{
    if (this != &hdr)
    {
        move_from(move(hdr));
        hdr.Clear();
    }
    return *this;
}

CBlockHeader::CBlockHeader(const CBlockHeader& hdr) noexcept
{
    copy_from(hdr);
}

CBlockHeader& CBlockHeader::operator=(const CBlockHeader& hdr) noexcept
{
    if (this != &hdr)
        copy_from(hdr);
    return *this;
}

CBlockHeader& CBlockHeader::copy_from(const CBlockHeader& hdr) noexcept
{
    nVersion = hdr.nVersion;
    hashPrevBlock = hdr.hashPrevBlock;
    hashMerkleRoot = hdr.hashMerkleRoot;
    hashFinalSaplingRoot = hdr.hashFinalSaplingRoot;
    nTime = hdr.nTime;
    nBits = hdr.nBits;
    nNonce = hdr.nNonce;
    nSolution = hdr.nSolution;
    sPastelID = hdr.sPastelID;
    prevMerkleRootSignature = hdr.prevMerkleRootSignature;
    return *this;
}

CBlockHeader& CBlockHeader::move_from(CBlockHeader&& hdr) noexcept
{
    nVersion = hdr.nVersion;
    hashPrevBlock = move(hdr.hashPrevBlock);
    hashMerkleRoot = move(hdr.hashMerkleRoot);
    hashFinalSaplingRoot = move(hdr.hashFinalSaplingRoot);
    nTime = hdr.nTime;
    nBits = hdr.nBits;
    nNonce = move(hdr.nNonce);
    nSolution = move(hdr.nSolution);
    sPastelID = move(hdr.sPastelID);
    prevMerkleRootSignature = move(hdr.prevMerkleRootSignature);
    return *this;
}
