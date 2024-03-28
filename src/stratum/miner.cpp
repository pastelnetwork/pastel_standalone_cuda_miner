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

void miner(CStratumClient &client)
{
    using eh_type = Eh200_9;
    // constexpr size_t numHashes = numBlocks * eh.IndicesPerHashOutput;
    // constexpr size_t numSlots = numHashes / NSLOTS;

    // Allocate device memory for blake2b state
    auto devState = make_cuda_unique<blake2b_state>(1);
    // Allocate device memory for hash values
    auto devHashes = make_cuda_unique<uint32_t>(eh_type::NHashes * eh_type::HashWords);
    // Allocate device memory for XORed hash values
    auto devXoredHashes = make_cuda_unique<uint32_t>(eh_type::NSlots * eh_type::HashWords);
    // Allocate device memory for slot bitmaps
    auto devSlotBitmaps = make_cuda_unique<uint32_t>(eh_type::NSlots * (eh_type::NSlots / 32));
    // Allocate device memory for solutions and solution count
    auto devSolutions = make_cuda_unique<eh_type::solution>(MAXSOLUTIONS);
    auto devSolutionCount = make_cuda_unique<uint32_t>(1);

    vector<eh_type::solution> vHostSolutions;

    // initialize extra nonce with random value
    uint32_t nExtraNonce2_Start = random_uint32();
    const uint32_t extraNonce1 = client.getExtraNonce1();

    // Initialize and update the first blake2b_state
    blake2b_state hostInitialState;
    string sPersString = client.getPersString();
    unsigned char personalization[BLAKE2B_PERSONALBYTES] = {0};
    memcpy(personalization, sPersString.c_str(), sPersString.size());
    const uint32_t le_N = htole32(client.getN());
    const uint32_t le_K = htole32(client.getK());
    const auto p = &personalization[sPersString.size()];
    memcpy(p, &le_N, sizeof(le_N));
    memcpy(p + sizeof(le_N), &le_K, sizeof(le_K));
    blake2b_init_salt_personal_host(&hostInitialState, nullptr, 0, BLAKE2B_OUTBYTES, nullptr, 0, personalization, BLAKE2B_PERSONALBYTES);
    v_uint8 vEquihashInput = client.getEquihashInput();
    blake2b_update_host(&hostInitialState, vEquihashInput.data(), vEquihashInput.size());

    uint32_t nExtraNonce2 = nExtraNonce2_Start;

    // Mining loop
    while (true)
    {
        blake2b_state currState = hostInitialState;
        const uint256 &nonce = client.generateNonce(++nExtraNonce2);
        blake2b_update_host(&currState, nonce.begin(), nonce.size());

        // Copy blake2b states from host to the device
        copyToDevice(devState.get(), &currState, sizeof(currState));

        constexpr uint32_t threadsPerBlock = 256;
        // Generate initial hash values
        generateInitialHashes<eh_type>(devState.get(), devHashes.get(), threadsPerBlock);
        
        // Perform K rounds of collision detection and XORing
        for (uint32_t round = 0; round < client.getK(); round++)
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

        copySolutionsToHost<eh_type>(devSolutions.get(), nSolutionCount, vHostSolutions);
        
        // Process the solutions and store them in the result solutions vector
        // Submit the solutions to the pool
        for (const auto& solution : vHostSolutions)
        {
            // Construct the block header using the solution indices
            string sHexSolution = HexStr(solution.indices, solution.indices + eh_type::ProofSize);
            client.submitSolution(nExtraNonce2, HexStr(client.getTime()), client.getNonce().ToString(), sHexSolution);
        }

        // Check for new job notifications
    }
}

        // // Generate hashes
        // uint32_t startBlock = 0;
        // uint32_t endBlock = numBlocks;
        // constexpr uint32_t threadsPerBlock = 256;
        // uint32_t hashesPerBlock = HASHESPERBLAKE;

        // launchGenHashesKernel(devState.get(), startBlock, endBlock, threadsPerBlock, hashesPerBlock, devHashes.get());

        // // XOR hashes and store index trees
        // uint32_t numHashesXOR = numHashes;
        // constexpr uint32_t threadsPerBlockXOR = 256;

        // launchXorHashesKernel(devHashes.get(), devXoredHashes.get(), devIndexes.get(), numHashesXOR, threadsPerBlockXOR);

        // // Propagate XOR values up the tree
        // uint32_t numNodesTree = numSlots * 2 - 1;
        // uint32_t numBlocksTree = numSlots;

        // launchPropagateXorKernel(devSlotBitmaps.get(), devXoredHashes.get(), numNodesTree, numBlocksTree);

        // // Find solutions
        // uint32_t numHashesSol = numHashes;
        // uint32_t threadsPerBlockSol = 256;
        // uint32_t maxSolsPerThread = 4;

        // launchFindSolutionsKernel(devXoredHashes.get(), devIndexes.get(), devSolutions.get(), numHashesSol, threadsPerBlockSol, maxSolsPerThread);

        // // Perform final round of solution identification
        // uint32_t numHashesFinal = numHashes;
        // uint32_t maxSolutionsFinal = maxSolutions;

        // launchFinalSolutionKernel(devXoredHashes.get(), devIndexes.get(), devSolutions.get(), numHashesFinal, maxSolutionsFinal);

        // // Copy solutions from device to host
        // uint32_t* hostSolutions = new uint32_t[maxSolutions];
        // copySolutionsToHost(hostSolutions, devSolutions, maxSolutions);

        // // Process and submit solutions
        // for (size_t i = 0; i < maxSolutions; i += 2)
        // {
        //     if (hostSolutions[i] == 0 && hostSolutions[i + 1] == 0)
        //         break;

        //     // Construct the block header using the solution indices
        //     v_uint32 solutionIndices;
        //     solutionIndices.push_back(hostSolutions[i]);
        //     solutionIndices.push_back(hostSolutions[i + 1]);

        //     string solution = GetHexStringFromIndices(solutionIndices);
        //     string time = ntime;
        //     string nonceStr = to_string(nonce);

        //     string blockHeader = constructBlockHeader(header, time, nonceStr, solution);

        //     // Submit the solution to the pool
        //     // submitSolution(sockfd, jobId.c_str(), "worker1", blockHeader.c_str());

        //     delete[] hostSolutions;
        // }
