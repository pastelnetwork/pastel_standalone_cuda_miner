// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>

#include <local_types.h>

using eh_index = uint32_t;
using eh_trunc = uint8_t;

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

    // The number of 32-bit words needed to store the hash output
    static inline constexpr uint32_t HashWords = (WN / 8 + sizeof(uint32_t) - 1) / sizeof(uint32_t); // HASHWORDS=7
    // The number of bits used to represent a single digit in the Equihash solution
    static inline constexpr uint32_t CollisionBitLength = N / (K + 1); // DIGITBITS=20
    // The number of bytes used to store a single digit of the collision bits.
    static inline constexpr uint32_t CollisionByteLength = (CollisionBitLength + 7) / 8;
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
    static inline constexpr uint32_t NHashes = 2 * Base; // 2'097'152
    // the total number of hashes words required for the equihash algorithm
    static inline constexpr uint32_t NHashWords = NHashes * HashWords;
    // the number of slots in the equihash algorithm
    static inline constexpr uint32_t NSlots = Base; // 1'048'576
    // the number of 32-bit slot bitmaps
    static inline constexpr uint32_t NSlotBitmaps = NSlots / 32;
    static inline constexpr uint32_t NBucketSize = UINT16_MAX; // 65'535
    static inline constexpr uint32_t NBucketCount = (NHashes + NBucketSize - 1) / NBucketSize; // 33
    static inline constexpr uint32_t NHashStorageCount = NBucketCount * NBucketSize; // 2'147'680
    static inline constexpr uint32_t NHashStorageWords = NHashStorageCount * HashWords; // 150'236'160
    static inline constexpr uint32_t OverflowBucketIndex = NBucketCount - 1; // 32

    using solution_device_type = struct
    {
        eh_index indices[ProofSize];
    };
    using solution_type = v_uint32;
};

using Eh200_9 = Equihash<200, 9>;