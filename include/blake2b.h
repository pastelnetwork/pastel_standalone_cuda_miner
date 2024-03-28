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

// blake2b_state struct
typedef struct {
    uint64_t h[8]; // 32 bytes
    uint64_t t[2]; // 16 bytes
    uint64_t f[2]; // 16 bytes
    uint8_t  buf[BLAKE2B_BLOCKBYTES]; // 128 bytes
    size_t   buflen; // 8 bytes 
    size_t   outlen; // 8 bytes
    uint8_t  last_node; // 1 byte 
} blake2b_state;
