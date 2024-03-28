// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>

#include <blake2b.h>
#include <src/equihash/equihash.h>
#include <src/equihash/blake2b_host.h>

using namespace std;


template<unsigned int N, unsigned int K>
bool Equihash<N, K>::InitializeState(blake2b_state &state, const string &sPersString)
{
    if (sPersString.length() != Equihash<N, K>::PERS_STRING_LENGTH)
        return false;
    unsigned char personalization[BLAKE2B_PERSONALBYTES] = {0};
    memcpy(personalization, sPersString.c_str(), sPersString.length());
    const uint32_t le_N = htole32(WN);
    const uint32_t le_K = htole32(WK);
    const auto p = &personalization[sPersString.length()];
    memcpy(p, &le_N, sizeof(le_N));
    memcpy(p + sizeof(le_N), &le_K, sizeof(le_K));
    blake2b_init_salt_personal_host(&state, nullptr, 0, BLAKE2B_OUTBYTES, nullptr, 0, personalization, BLAKE2B_PERSONALBYTES);
    return true;
}

template<unsigned int N, unsigned int K>
bool IsValidSolution(const blake2b_state& base_state, const v_uint8 &soln)
{
    return false;
}

template class Equihash<200, 9>;