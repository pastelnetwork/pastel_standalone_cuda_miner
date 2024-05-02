// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <cstddef>

constexpr size_t BLAKE2B_BLOCKBYTES = 128;
constexpr size_t BLAKE2B_OUTBYTES = 64;
constexpr size_t BLAKE2B_KEYBYTES = 64;
constexpr size_t BLAKE2B_SALTBYTES = 16;
constexpr size_t BLAKE2B_PERSONALBYTES = 16;

#ifdef _MSC_VER
#pragma pack(push, 1)
#elif defined(__GNUC__) || defined(__clang__)
#pragma pack(1)
#endif

// blake2b_state struct
typedef struct blake2b_state_
{
    uint64_t h[8]; // 64 bytes
    uint64_t t[2]; // 16 bytes
    uint64_t f[2]; // 16 bytes
    uint8_t  buf[2 * BLAKE2B_BLOCKBYTES]; // 256 bytes
    size_t   buflen; // 8 bytes 
    uint8_t  last_node; // 1 byte 
} blake2b_state;

typedef struct blake2b_param_
{
    uint8_t digest_length;                   /*  1 */
    uint8_t key_length;                      /*  2 */
    uint8_t fanout;                          /*  3 */
    uint8_t depth;                           /*  4 */
    uint8_t leaf_length[4];                  /*  8 */
    uint8_t node_offset[8];                  /* 16 */
    uint8_t node_depth;                      /* 17 */
    uint8_t inner_length;                    /* 18 */
    uint8_t reserved[14];                    /* 32 */
    uint8_t salt[BLAKE2B_SALTBYTES];         /* 48 */
    uint8_t personal[BLAKE2B_PERSONALBYTES]; /* 64 */
} blake2b_param;

#ifdef _MSC_VER
#pragma pack(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma pack()
#endif
