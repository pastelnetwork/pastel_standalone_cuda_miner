// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <string>
#include <functional>
#include <vector>

#include <blake2b.h>
#include <src/equihash/equihash-types.h>
#include <src/utils/uint256.h>
#include <src/utils/svc_thread.h>
#include <src/stratum/client.h>

class CMiningThread : public CStoppableServiceThread
{
public:
        using EquihashType = Eh200_9; 

        CMiningThread(CStratumClient &StratumClient);

        void execute() override;

        virtual uint256 generateNonce(const uint32_t nExtraNonce2) const noexcept;
        virtual void submitSolution(const uint32_t nExtraNonce2, const std::string& sTime, const std::string &sHexSolution);

protected:
        CStratumClient &m_StratumClient;

        uint32_t miningLoop(const blake2b_state& initialState, uint32_t &nExtraNonce2, const std::string &sTime,
                const size_t nIterations, const uint32_t threadsPerBlock);
};

