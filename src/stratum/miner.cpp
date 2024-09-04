// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <src/stratum/miner.h>
#include <src/utils/random.h>
#include <src/utils/strencodings.h>
#include <src/equihash/equihash.h>
#include <src/equihash/equihash-helper.h>
#include <src/equihash/blake2b_host.h>
#include <src/stratum/client.h>
#include <local_types.h>
#include <src/kernel/memutils.h>
#include <src/kernel/kernel.h>

using namespace std;

CMiningThread::CMiningThread(CStratumClient &StratumClient) : 
    CStoppableServiceThread("MiningThread"),
    m_StratumClient(StratumClient)
{}

void CMiningThread::execute()
{
    // initialize extra nonce with random value
    uint32_t nExtraNonce2_Start = random_uint32();
    const uint32_t extraNonce1 = m_StratumClient.getExtraNonce1();
    uint32_t nExtraNonce2 = nExtraNonce2_Start;

    // Initialize and update the first blake2b_state
    string sPersString = m_StratumClient.getPersString();

    using eh_type = EquihashSolver<200, 9>;
    auto eh = eh_type();
    blake2b_state state;
    eh.InitializeState(state, sPersString);
    v_uint8 vEquihashInput = m_StratumClient.getEquihashInput();
    blake2b_update_host(&state, vEquihashInput.data(), vEquihashInput.size());

    constexpr uint32_t threadsPerBlock = 256;

    miningLoop(state, nExtraNonce2, HexStr(m_StratumClient.getTime()), 100, threadsPerBlock);
}

uint256 CMiningThread::generateNonce(const uint32_t nExtraNonce2) const noexcept
{
    return m_StratumClient.generateNonce(nExtraNonce2);
}

void CMiningThread::submitSolution(const uint32_t nExtraNonce2, const string& sTime, const string &sHexSolution)
{
    m_StratumClient.submitSolution(nExtraNonce2, sTime, sHexSolution);
}

uint32_t CMiningThread::miningLoop(const blake2b_state& initialState, uint32_t &nExtraNonce2, const string &sTime,
                    const size_t nIterations, const uint32_t threadsPerBlock)
{
    EhDevice<EquihashType> devStore;
    auto eh = EquihashSolver<EquihashType::WN, EquihashType::WK>();

    vector<typename EquihashType::solution_type> vHostSolutions;
    uint32_t nTotalSolutionCount = 0;

    for (uint32_t i = 0; i < nIterations; ++i)
    {
        if (shouldStop())
            break;

        blake2b_state currState = initialState;
        const uint256 nonce = generateNonce(nExtraNonce2);
        blake2b_update_host(&currState, nonce.begin(), nonce.size());

        // Copy blake2b states from host to the device
        copyToDevice(devStore.initialState.get(), &currState, sizeof(currState));

        const uint32_t nSolutionCount = devStore.solver();
        
        nTotalSolutionCount += nSolutionCount;
        if (nSolutionCount == 0)
        {
            cout << "No solutions found for extra nonce 2: " << nExtraNonce2 << endl;
            ++nExtraNonce2;
            continue;
        }

        devStore.copySolutionsToHost(vHostSolutions);

        DBG_EQUI_WRITE_FN(devStore.debugWriteSolutions(vHostSolutions));
        //devStore.debugTraceSolution(1000);

        // check solutions
        v_uint8 solutionMinimal;
        v_uint32 vSolution;
        vSolution.resize(EquihashType::ProofSize);
        string sError;
        size_t num = 1;
        for (const auto& solution : vHostSolutions)
        {
            memcpy(vSolution.data(), &solution.indices, sizeof(solution.indices));
            // string s;
            // for (auto i : vSolution)
            //     s += to_string(i) + " ";
            // cout << "new solution indices:" << s << endl;

            solutionMinimal = GetMinimalFromIndices(vSolution, EquihashType::CollisionBitLength);
            if (!eh.IsValidSolution(sError, currState, solutionMinimal))
            {
                cerr << dec << num << ": invalid solution: " << sError << endl;
                continue;
            } else
            {
                cout << strprintf("%zu: valid solution", num) << endl;
            }

            string sHexSolution = HexStr(solutionMinimal);
            submitSolution(nExtraNonce2, sTime, sHexSolution);
            ++num;

            if (shouldStop())
                break;
        }

        ++nExtraNonce2;
    }

    return nTotalSolutionCount;
}
