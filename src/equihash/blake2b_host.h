// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstddef>

#include <blake2b.h>

bool blake2b_init_host(blake2b_state *S, size_t outlen);
void blake2b_update_host(blake2b_state *S, const void *pin, size_t inlen);
bool blake2b_final_host(blake2b_state *S, uint8_t *out, const size_t outlen);

bool blake2b_init_salt_personal_host(blake2b_state *state, 
    const uint8_t* key, size_t nKeyLength, 
    size_t outlen,
    const uint8_t* salt, const size_t nSaltLength,
    const uint8_t* personal, const size_t nPersonaLength);
