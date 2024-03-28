// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstdint>
#include <string>

#include <local_types.h>
#include <src/utils/uint256.h>
#include <src/utils/serialize.h>

class CBlockHeader
{
public:
    // block header
    int32_t nVersion;             // version of the block
    uint256 hashPrevBlock;        // hash of the previous block
    uint256 hashMerkleRoot;       // merkle root
    uint256 hashFinalSaplingRoot; // final sapling root (hash representing a state of the Sapling shielded transactions)
    uint32_t nTime;			      // Unix timestamp of the block (when the miner started hashing the header)
    uint32_t nBits;			      // difficulty of the proof of work (target threshold for the block's hash)
    uint256 nNonce;			      // 256-bit number that miners change to modify the header hash 
    // in order to produce a hash below the target threshold (nBits)
    v_uint8 nSolution;            // Equihash solution - can be empty vector
    // v5
    std::string sPastelID;        // mnid of the SN that mined the block (public key to verify signature)
    v_uint8 prevMerkleRootSignature; // signature for the merkle root hash of the previous block signed with the SN private key

    CBlockHeader() noexcept = default;
    virtual ~CBlockHeader() noexcept = default;

    void Clear() noexcept
    {
        nVersion = CBlockHeader::CURRENT_VERSION;
        hashPrevBlock.SetNull();
        hashMerkleRoot.SetNull();
        hashFinalSaplingRoot.SetNull();
        nTime = 0;
        nBits = 0;
        nNonce.SetNull();
        nSolution.clear();
        sPastelID.clear();
        prevMerkleRootSignature.clear();
    }

    bool IsNull() const noexcept { return (nBits == 0); }

    // current version of the block header
    static constexpr int32_t CURRENT_VERSION = 5;
    static constexpr int32_t VERSION_CANONICAL = 4;
    static constexpr int32_t VERSION_SIGNED_BLOCK = 5;
};

/**
 * Custom serializer for CBlockHeader that omits the nonce and solution, for use
 * as input to Equihash.
 */
class CEquihashInput : private CBlockHeader
{
public:
    CEquihashInput(const CBlockHeader &header)
    {
        CBlockHeader::Clear();
        *((CBlockHeader*)this) = header;
    }

    ADD_SERIALIZE_METHODS;

    template <typename Stream>
    inline void SerializationOp(Stream& s, const SERIALIZE_ACTION ser_action)
    {
        READWRITE(this->nVersion);
        READWRITE(hashPrevBlock);
        READWRITE(hashMerkleRoot);
        READWRITE(hashFinalSaplingRoot);
        READWRITE(nTime);
        READWRITE(nBits);
        if (nVersion >= CBlockHeader::VERSION_SIGNED_BLOCK)
        {
            READWRITE_CHECKED(sPastelID, 100);
            READWRITE_CHECKED(prevMerkleRootSignature, 200);
        }
    }

    constexpr size_t GetReserveSize() const noexcept
    {
        size_t nReserveSize = sizeof(nVersion) +
            sizeof(hashPrevBlock) +
            sizeof(hashMerkleRoot) +
            sizeof(hashFinalSaplingRoot) +
            sizeof(nTime) +
            sizeof(nBits);
        if (nVersion >= CBlockHeader::VERSION_SIGNED_BLOCK)
        {
            nReserveSize +=
                87 + // 86-bytes Pastel ID + 1-byte size
                115;  // 114-bytes prev merkle root signature + 1-byte size
        }
        return nReserveSize;
	}
};
