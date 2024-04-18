// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <vector>
#include <endian.h>

#include <tinyformat.h>
#include <blake2b.h>
#include <src/equihash/equihash.h>
#include <src/equihash/blake2b_host.h>
#include <src/equihash/equihash-types.h>
#include <src/equihash/equihash-helper.h>

using namespace std;

template<unsigned int N, unsigned int K>
bool EquihashSolver<N, K>::InitializeState(blake2b_state &state, const string &sPersString)
{
    if (sPersString.length() != Equihash<N, K>::PERS_STRING_LENGTH)
        return false;
    uint8_t personalization[BLAKE2B_PERSONALBYTES] = {0};
    memcpy(personalization, sPersString.c_str(), sPersString.length());
    const uint32_t constN = Equihash<N, K>::WN;
    const uint32_t constK = Equihash<N, K>::WK;
    const uint32_t le_N = htole32(constN);
    const uint32_t le_K = htole32(constK);
    const auto p = &personalization[sPersString.length()];
    memcpy(p, &le_N, sizeof(le_N));
    memcpy(p + sizeof(le_N), &le_K, sizeof(le_K));
    blake2b_init_salt_personal_host(&state, nullptr, 0, Equihash<N, K>::HashOutput, nullptr, 0, personalization, sizeof(personalization));
    return true;
}

template<unsigned int N, unsigned int K>
bool EquihashSolver<N, K>::IsValidSolution(string &error, const blake2b_state& base_state, const v_uint8 &soln)
{
    if (soln.size() != Equihash<N, K>::SolutionWidth)
    {
        error = strprintf("Invalid solution length: %d (expected %d)\n",
                 soln.size(), Equihash<N, K>::SolutionWidth);
        return false;
    }

    vector<FullStepRow<Equihash<N, K>::FinalFullWidth>> X;
    X.reserve(Equihash<N, K>::ProofSize);
    uint8_t tmpHash[Equihash<N, K>::HashOutput];
    v_uint32 vIndices = GetIndicesFromMinimal(soln, Equihash<N, K>::CollisionBitLength);
    for (eh_index i : vIndices)
    {
        GenerateHash(base_state, i/Equihash<N, K>::IndicesPerHashOutput, tmpHash, sizeof(tmpHash));
        X.emplace_back(tmpHash+((i % Equihash<N, K>::IndicesPerHashOutput) * N/8),
                       N/8, Equihash<N, K>::HashLength, Equihash<N, K>::CollisionBitLength, i);
    }

    size_t hashLen = Equihash<N, K>::HashLength;
    size_t lenIndices = sizeof(eh_index);
    const int iColByteLength = static_cast<const int>(Equihash<N, K>::CollisionByteLength);
    while (X.size() > 1)
    {
        vector<FullStepRow<Equihash<N, K>::FinalFullWidth>> Xc;
        for (int i = 0; i < X.size(); i += 2)
        {
            const uint32_t indexLeft = vIndices[i];
            const uint32_t indexRight = vIndices[i + 1];
            if (!HasCollision(X[i], X[i + 1], Equihash<N, K>::CollisionByteLength))
            {
                error = strprintf(
R"(Invalid solution: invalid collision length between StepRows
X[%u] = %s,
X[%u] = %s)", indexLeft, X[i].GetHex(hashLen), indexRight, X[i + 1].GetHex(hashLen));
                return false;
            }
            if (X[i + 1].IndicesBefore(X[i], hashLen, lenIndices))
            {
                error = "Invalid solution: Index tree incorrectly ordered";
                return false;
            }
            if (!DistinctIndices(X[i], X[i + 1], hashLen, lenIndices))
            {
                error = "Invalid solution: duplicate indices";
                return false;
            }
            Xc.emplace_back(X[i], X[i + 1], hashLen, lenIndices, iColByteLength);
        }
        X = Xc;
        hashLen -= Equihash<N, K>::CollisionByteLength;
        lenIndices *= 2;
    }

    assert(X.size() == 1);
    return X[0].IsZero(hashLen);
}

template class FullStepRow<Eh200_9::FinalFullWidth>;
template class EquihashSolver<200, 9>;
