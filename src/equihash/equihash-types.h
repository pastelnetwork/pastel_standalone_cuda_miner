// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <array>
#include <limits>

#include <local_types.h>

using eh_index = uint32_t;
using eh_trunc = uint8_t;

inline constexpr int COUNT_BITS(unsigned int N)
{
    int bits = 0;
    while (N > 0)
    {
        bits++;
        N >>= 1;
    }
    return bits;
}

/**
 * The Equihash class provides a set of constants for the Equihash proof-of-work algorithm.
 * 
 * \param N - Represents the bit length of the hash output used in the puzzle. It determines the overall size of 
   the hashing problem and, indirectly, the amount of memory required to store the hash outputs generated during the solving process.
 * \param K -  work factor used in the proof-of-work algorithm. The number of hashing rounds and the complexity of finding a solution.
   It influences the number of hash pairs that must be XORed together to find a solution that meets the algorithm's requirements.
 */
template<unsigned int N, unsigned int K>
class Equihash
{
private:
    static_assert(K < N, "K must be less than N.");
    static_assert(N % 8 == 0, "N must be a multiple of 8.");
    static_assert((N / (K + 1)) + 1 < 8 * sizeof(eh_index), "Size of eh_index must be sufficient to represent indices.");

public:
    static inline constexpr uint32_t WN = N;
    static inline constexpr uint32_t WK = K;
    static inline constexpr uint32_t PERS_STRING_LENGTH = 8;

    // the number of indices in one equihash solution (2 ^ K)
    static inline constexpr uint32_t ProofSize = 1 << K; // PROOFSIZE=512
    // The number of hash outputs that can be indexed per each hash operation based on N.
     static inline constexpr uint32_t IndicesPerHashOutput = 512 / N;  // HASHESPERBLAKE=2
    // The output size of the blake2b hash in bytes required 
    static inline constexpr uint32_t HashOutput = IndicesPerHashOutput * N / 8; // HASHOUT=50
    static inline constexpr uint32_t SingleHashOutput = WN / 8; // SINGLEHASHOUT=25

    // The number of 32-bit words needed to store the hash output
    static inline constexpr uint32_t HashWords = (WN / 8 + sizeof(uint32_t) - 1) / sizeof(uint32_t); // HASHWORDS=7
    static inline constexpr uint32_t HashFullWords = SingleHashOutput / sizeof(uint32_t); // HASHFULLWORDS=6
    static inline constexpr uint32_t HashPartialBytesLeft = SingleHashOutput % sizeof(uint32_t); // HASHPARTIALBYTESLEFT=1
    // The number of bits used to represent a single digit in the Equihash solution
    static inline constexpr uint32_t CollisionBitLength = N / (K + 1); // DIGITBITS=20
    // The number of bytes used to store a single digit of the collision bits.
    static inline constexpr uint32_t CollisionByteLength = (CollisionBitLength + 7) / 8;
    static inline constexpr uint32_t CollisionBitMask = (1 << CollisionBitLength) - 1;
    static inline constexpr uint32_t CollisionBitMaskWordPadding = std::numeric_limits<uint32_t>::digits - CollisionBitLength; 
    // The length in bytes of a hash used during the (K+1)-th collision round.
    static inline constexpr uint32_t HashLength = (K + 1) * CollisionByteLength;
    // The full width in bytes of a list entry before the final round, including collision data and indices.
    static inline constexpr uint32_t FullWidth = 2 * CollisionByteLength + sizeof(eh_index) * (1 << (K - 1));
    // The full width in bytes of a list entry during the final round, including collision data and indices.
    static inline constexpr uint32_t FinalFullWidth = 2 * CollisionByteLength + sizeof(eh_index) * ProofSize;
    // The maximum width in bytes of a list entry before the final round when using truncated hash representations.
    static inline constexpr uint32_t TruncatedWidth = std::max(HashLength + sizeof(eh_trunc), 2 * CollisionByteLength+sizeof(eh_trunc) * (1 << (K - 1)));
    // The maximum width in bytes of a list entry during the final round when using truncated hash representations.
    static inline constexpr uint32_t FinalTruncatedWidth = std::max(HashLength+sizeof(eh_trunc), 2 * CollisionByteLength+sizeof(eh_trunc) * ProofSize);
    // The width in bytes of the serialized solution that satisfies the Equihash puzzle.
    static inline constexpr uint32_t SolutionWidth = ProofSize * (CollisionBitLength + 1) / 8; // SOLUTIONSIZE=1344

    // the base value used in the equihash algorithm
    static inline constexpr uint32_t Base = 1 << CollisionBitLength; // 1'048'576
    // the total number of hashes required for the equihash algorithm
    static inline constexpr uint32_t NHashes = IndicesPerHashOutput * Base; // 2'097'152
    // the total number of hashes words required for the equihash algorithm
    static inline constexpr uint32_t NHashWords = NHashes * HashWords; // 14'680'064
    static inline constexpr uint32_t NBucketCount = 2'048;
    static inline constexpr uint32_t NBucketCountStorageSize = NBucketCount * sizeof(uint32_t);
    static inline constexpr uint32_t NExtraHashesPerBucket = 126;
    static inline constexpr uint32_t NBucketSize = (NHashes + NBucketCount - 1) / NBucketCount; // 1'024
    static inline constexpr uint32_t NBucketSizeExtra = NBucketSize + NExtraHashesPerBucket; // 1'150
    static inline constexpr uint32_t NBucketSizeExtraBoolMaskSize = NBucketSizeExtra * sizeof(bool);
    static inline constexpr uint32_t NBucketIdxBits = COUNT_BITS(NBucketCount - 1); // 11
    static inline constexpr uint32_t NBucketIdxMask = NBucketCount - 1; // 2'047
    static inline constexpr uint32_t CollisionThreadsPerBlock = 256;
    static inline constexpr uint32_t NCollisionIndexBits = 11;
    static inline constexpr uint32_t NCollisionIndexBitMask = (1 << NCollisionIndexBits) - 1; // 2'047
    static inline constexpr uint32_t NHashStorageCount = NBucketCount * NBucketSizeExtra; // 2'048 * 1'150 = 2'355'200
    static inline constexpr uint32_t NHashStorageWords = NBucketCount * NBucketSizeExtra * HashWords; // 2'330'200 * 7 = 16'486'400

    // __shared__ uint64_t sharedProcessed[threadsPerBlock * maskArraySize];

    using solution_type = struct
    {
        eh_index indices[ProofSize];
    };

private:
    // Function to compute hash word offset for the given round
    static constexpr uint32_t computeHashWordOffset(const uint32_t round)
    {
        const uint32_t globalBitOffset = round * CollisionBitLength;
        uint32_t wordOffset = globalBitOffset / std::numeric_limits<uint32_t>::digits;
        if (wordOffset >= HashWords - 1)
            wordOffset = HashWords - 2;
        return wordOffset;
    }

    // Helper constexpr function to populate the hashWordOffsets array
    static constexpr std::array<uint32_t, WK> makeHashWordOffsets()
    {
        std::array<uint32_t, WK> offsets{};
        for (size_t i = 0; i < WK; ++i)
            offsets[i] = computeHashWordOffset(i);
        return offsets;
    }

    static constexpr uint32_t computeHashBitOffset(const uint32_t round)
    {
        const uint32_t globalBitOffset = round * CollisionBitLength;
        return globalBitOffset - computeHashWordOffset(round) * std::numeric_limits<uint32_t>::digits;
    }

    static constexpr std::array<uint32_t, WK> makeHashBitOffsets()
    {
        std::array<uint32_t, WK> offsets{};
        for (size_t i = 0; i < WK; ++i)
            offsets[i] = computeHashBitOffset(i);
        return offsets;
    }

/*
round: 0, wordOffset: 0, bitOffset  0, collisionBitMask: 00000000 00f0ffff || xoredWordOffset: 0, xoredBitOffset 16
round: 1, wordOffset: 0, bitOffset 20, collisionBitMask: 000000ff ff0f0000 || xoredWordOffset: 1, xoredBitOffset 8
round: 2, wordOffset: 1, bitOffset 08, collisionBitMask: 00000000 f0ffff00 || xoredWordOffset: 1, xoredBitOffset 24
round: 3, wordOffset: 1, bitOffset 28, collisionBitMask: 0000ffff 0f000000 || xoredWordOffset: 2, xoredBitOffset 16
round: 4, wordOffset: 2, bitOffset 16, collisionBitMask: 000000f0 ffff0000 || xoredWordOffset: 3, xoredBitOffset 0
round: 5, wordOffset: 3, bitOffset 04, collisionBitMask: 00000000 00ffff0f || xoredWordOffset: 3, xoredBitOffset 24
round: 6, wordOffset: 3, bitOffset 24, collisionBitMask: 0000f0ff ff000000 || xoredWordOffset: 4, xoredBitOffset 8
round: 7, wordOffset: 4, bitOffset 12, collisionBitMask: 00000000 ffff0f00 || xoredWordOffset: 5, xoredBitOffset 0
round: 8, wordOffset: 5, bitOffset 00, collisionBitMask: 00000000 00f0ffff || xoredWordOffset: 5, xoredBitOffset 16
*/
    static constexpr std::array<uint64_t, WK> makeHashCollisionMasks()
    {
        std::array<uint64_t, WK> masks{};
        constexpr bool bNeedAdditionalByte = CollisionBitLength % 8 != 0;
        uint32_t fullByteBitMask = (1U << (CollisionBitLength / 8) * 8) - 1;  // collision mask for full bytes
        for (size_t i = 0; i < WK; ++i)
        {
            uint32_t bitOffset = computeHashBitOffset(i);
            uint64_t mask = 0;
            if constexpr (bNeedAdditionalByte)
            {
                // Adjust the mask based on the bit offset's alignment
                if (bitOffset % 8 == 4) // Odd alignment, nibble starts in the middle
                {
                    mask = (fullByteBitMask << 8) | 0xFULL;
                    bitOffset -= 4;
                } else // Even alignment
                    mask = (0xF0ULL << (CollisionBitLength - 4)) | fullByteBitMask;
            } else
                mask = fullByteBitMask;
            masks[i] = mask << bitOffset;
        }
        return masks;
    }

    static constexpr uint32_t computeXoredHashWordOffset(const uint32_t round)
    {
        const uint32_t xoredGlobalBitOffset = (round + 1) * CollisionBitLength;
        uint32_t xoredWordOffset = xoredGlobalBitOffset / std::numeric_limits<uint32_t>::digits;
        if (xoredWordOffset >= HashWords - 1)
            xoredWordOffset = HashWords - 2;
        return xoredWordOffset;
    }

    static constexpr std::array<uint32_t, WK> makeXoredHashWordOffsets()
    {
        std::array<uint32_t, WK> offsets{};
        for (size_t i = 0; i < WK; ++i)
            offsets[i] = computeXoredHashWordOffset(i);
        return offsets;
    }

    static constexpr uint32_t computeXoredHashBitOffset(const uint32_t round)
    {
        const uint32_t xoredGlobalBitOffset = (round + 1) * CollisionBitLength;
        uint32_t xoredBitOffset = xoredGlobalBitOffset - computeXoredHashWordOffset(round) * std::numeric_limits<uint32_t>::digits;
        if (xoredBitOffset % 8 == 4)
            xoredBitOffset -= 4;
        return xoredBitOffset;
    }

    static constexpr std::array<uint32_t, WK> makeXoredHashBitOffsets()
    {
        std::array<uint32_t, WK> offsets{};
        for (size_t i = 0; i < WK; ++i)
            offsets[i] = computeXoredHashBitOffset(i);
        return offsets;
    }

public:
    // Array to store hash word offsets for each round
    static inline constexpr std::array<uint32_t, WK> HashWordOffsets = makeHashWordOffsets();
    // Array to store hash bit offsets for each round
    static inline constexpr std::array<uint32_t, WK> HashBitOffsets = makeHashBitOffsets();
    // Array to store hash collision masks for each round
    static inline constexpr std::array<uint64_t, WK> HashCollisionMasks = makeHashCollisionMasks();
    // Array to store xored hash word offsets for each round
    static inline constexpr std::array<uint32_t, WK> XoredHashWordOffsets = makeXoredHashWordOffsets();
    // Array to store xored hash bit offsets for each round
    static inline constexpr std::array<uint32_t, WK> XoredHashBitOffsets = makeXoredHashBitOffsets();
};

using Eh200_9 = Equihash<200, 9>;