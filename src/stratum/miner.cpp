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

// Function to submit a solution to the pool
void submitSolution(int sockfd, const char* jobId, const char* workerName, const char* solution)
{
    // Construct the mining.submit message
    string submitMsg = "{\"id\": 1, \"method\": \"mining.submit\", \"params\": [\"" +
                            string(workerName) + "\", \"" + string(jobId) + "\", \"" +
                            string(solution) + "\"]}\n";

    // Send the message to the pool
    if (send(sockfd, submitMsg.c_str(), submitMsg.length(), 0) < 0) {
        cerr << "Error sending solution to pool" << endl;
    }
}

template<typename EquihashType>
uint32_t miningLoop(const blake2b_state& initialState, uint32_t &nExtraNonce2, const string &sTime,
                    const size_t nIterations, const uint32_t threadsPerBlock,
                    const funcGenerateNonce_t &genNonceFn, const funcSubmitSolution_t &submitSolutionFn)
{
    using eh_type = EquihashType;

    // Allocate device memory for blake2b state
    auto devState = make_cuda_unique<blake2b_state>(1);
    // Allocate device memory for hash values
    auto devHashes = make_cuda_unique<uint32_t>(eh_type::NHashes * eh_type::HashWords);
    // Allocate device memory for XORed hash values
    auto devXoredHashes = make_cuda_unique<uint32_t>(eh_type::NSlots * eh_type::HashWords);
    // Allocate device memory for slot bitmaps
    auto devSlotBitmaps = make_cuda_unique<uint32_t>(eh_type::NSlots * (eh_type::NSlots / 32));
    // Allocate device memory for solutions and solution count
    auto devSolutions = make_cuda_unique<typename eh_type::solution>(MAXSOLUTIONS);
    auto devSolutionCount = make_cuda_unique<uint32_t>(1);

    vector<typename eh_type::solution> vHostSolutions;
    uint32_t nTotalSolutionCount = 0;

    for (uint32_t i = 0; i < nIterations; ++i)
    {
        blake2b_state currState = initialState;
        const uint256 nonce = genNonceFn(nExtraNonce2);
        blake2b_update_host(&currState, nonce.begin(), nonce.size());

        // Copy blake2b states from host to the device
        copyToDevice(devState.get(), &currState, sizeof(currState));

        // Generate initial hash values
        generateInitialHashes<eh_type>(devState.get(), devHashes.get(), threadsPerBlock);

        // Perform K rounds of collision detection and XORing
        for (uint32_t round = 0; round < EquihashType::WK; round++)
        {
            // Detect collisions and XOR the colliding pairs
            detectCollisions<eh_type>(devHashes.get(), devSlotBitmaps.get(), threadsPerBlock);
            xorCollisions<eh_type>(devHashes.get(), devSlotBitmaps.get(), devXoredHashes.get(), threadsPerBlock);

            // Swap the hash pointers for the next round
            swap(devHashes, devXoredHashes);
        }

        // Find valid solutions
        const uint32_t nSolutionCount = findSolutions<eh_type>(devHashes.get(), devSlotBitmaps.get(),
            devSolutions.get(), devSolutionCount.get(), threadsPerBlock);

        nTotalSolutionCount += nSolutionCount;

        copySolutionsToHost<eh_type>(devSolutions.get(), nSolutionCount, vHostSolutions);

        // Process the solutions and submit them
        for (const auto& solution : vHostSolutions)
        {
            string sHexSolution = HexStr(solution.indices, solution.indices + eh_type::ProofSize);
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
