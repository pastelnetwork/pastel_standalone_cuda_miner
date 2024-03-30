// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <src/stratum/miner.h>
#include <src/utils/random.h>
#include <src/utils/strencodings.h>
#include <src/equihash/equihash.h>
#include <src/equihash/blake2b_host.h>
#include <local_types.h>
#include <src/cuda/memutils.h>
#include <src/cuda/kernel.h>

using namespace std;

template<typename EquihashType>
uint32_t miningLoop(const blake2b_state& initialState, uint32_t &nExtraNonce2, const string &sTime,
                    const size_t nIterations, const uint32_t threadsPerBlock,
                    const funcGenerateNonce_t &genNonceFn, const funcSubmitSolution_t &submitSolutionFn)
{
    EhDeviceStore<EquihashType> devStore;
    if (!devStore.allocate_memory())
    {
        cerr << "Failed to allocate CUDA memory for Equihash solver" << endl;
        return 0;
    }    

    vector<typename EquihashType::solution> vHostSolutions;
    uint32_t nTotalSolutionCount = 0;

    for (uint32_t i = 0; i < nIterations; ++i)
    {
        blake2b_state currState = initialState;
        const uint256 nonce = genNonceFn(nExtraNonce2++);
        blake2b_update_host(&currState, nonce.begin(), nonce.size());

        // Copy blake2b states from host to the device
        copyToDevice(devStore.initialState.get(), &currState, sizeof(currState));

        // Generate initial hash values
        generateInitialHashes<EquihashType>(devStore);

// debug
v_uint32 hostHashes(EquihashType::NHashes * EquihashType::HashWords);
copyToHost(hostHashes.data(), devHashes.get(), hostHashes.size() * sizeof(uint32_t));

// Print out the generated hashes
size_t hashNo = 0;
for (size_t i = 0; i < hostHashes.size(); i += EquihashType::HashWords)
{
    ++hashNo;
    if (hashNo % 0x1000 != 0)
        continue;
    cout << "Hash " << dec << hashNo << ": ";
    bool bAllZeroes = true;
    for (size_t j = 0; j < EquihashType::HashWords; ++j)
    {
        if (hostHashes[i + j])
            bAllZeroes = false;
        cout << hex << hostHashes[i + j] << " ";
    }
    if (bAllZeroes)
    {
        cout << "All zeroes" << endl;
        break;
    }
    cout << endl;
}
// debug

        // Perform K rounds of collision detection and XORing
        for (uint32_t round = 0; round < EquihashType::WK; round++)
        {
            // Detect collisions and XOR the colliding pairs
            detectCollisions<EquihashType>(devHashes.get(), devSlotBitmaps.get(), threadsPerBlock);
            xorCollisions<EquihashType>(devHashes.get(), devSlotBitmaps.get(), devXoredHashes.get(), threadsPerBlock);

            // Swap the hash pointers for the next round
            swap(devHashes, devXoredHashes);
        }

        // Find valid solutions
        const uint32_t nSolutionCount = findSolutions<EquihashType>(devHashes.get(), devSlotBitmaps.get(),
            devSolutions.get(), devSolutionCount.get(), threadsPerBlock);

        nTotalSolutionCount += nSolutionCount;

        if (nSolutionCount > 0)
            copySolutionsToHost<EquihashType>(devSolutions.get(), nSolutionCount, vHostSolutions);

        // Process the solutions and submit them
        for (const auto& solution : vHostSolutions)
        {
            string sHexSolution = HexStr(solution.indices, solution.indices + EquihashType::ProofSize);
            submitSolutionFn(nExtraNonce2, sTime, nonce.GetHex(), sHexSolution);
        }
    }

    return nTotalSolutionCount;
}

void miner(CStratumClient &client)
{
    // initialize extra nonce with random value
    uint32_t nExtraNonce2_Start = random_uint32();
    const uint32_t extraNonce1 = client.getExtraNonce1();
    uint32_t nExtraNonce2 = nExtraNonce2_Start;

    // Initialize and update the first blake2b_state
    string sPersString = client.getPersString();

    using eh_type = Eh200_9;
    auto eh = eh_type();
    blake2b_state state;
    eh.InitializeState(state, sPersString);
    v_uint8 vEquihashInput = client.getEquihashInput();
    blake2b_update_host(&state, vEquihashInput.data(), vEquihashInput.size());

    const auto generateNonceFn = [&client](uint32_t nExtraNonce2) -> const uint256
    {
        return client.generateNonce(nExtraNonce2);
    };

    const auto submitSolutionFn = [&client](const uint32_t nExtraNonce2, const string& sTime, 
        const string& sNonce, const string &sHexSolution)
    {
        client.submitSolution(nExtraNonce2, sTime, sNonce, sHexSolution);
    };
    constexpr uint32_t threadsPerBlock = 256;

    miningLoop<eh_type>(state, nExtraNonce2, HexStr(client.getTime()), 1, threadsPerBlock, generateNonceFn, submitSolutionFn);
}
