// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <string>
#include <functional>
#include <vector>
#include <chrono>
#include <atomic>

#include <blake2b.h>
#include <src/utils/uint256.h>
#include <src/utils/svc_thread.h>
#include <src/utils/streams.h>
#include <src/equihash/equihash-types.h>
#include <src/equihash/equihash.h>
#include <src/equihash/block.h>
#include <src/kernel/kernel.h>
#include <src/stratum/client_intf.h>

class CMiningThread : public CStoppableServiceThread
{
public:
    using EquihashType = Eh200_9; 
    using EquihashSolverType = EquihashSolver<EquihashType::WN, EquihashType::WK>;

    CMiningThread(IStratumClient *pStratumClient);

    void execute() override;
    virtual uint256 generateNonce(const uint32_t nExtraNonce2) noexcept;
    virtual bool submitSolution(const std::string& sNonce2, const std::string& sTime, const std::string &sHexSolution);
	void NewJobNotify();
	void StopCurrentJob();
    virtual uint32_t miningLoop(const size_t nIterations);
	virtual void AssignNewJob();

    double getJobSolPs() const noexcept;
    double getTotalSolPs() const noexcept;
	CEquihashInput& getBlockHeader() noexcept { return m_blockHeader; }
	const blake2b_state& getInitialState() const noexcept { return m_initialState; }

	void setExtraNonce1(const uint32_t nExtraNonce1) noexcept { m_nExtraNonce1 = nExtraNonce1; }

protected:
    CEquihashInput m_blockHeader;
	IStratumClient *m_pStratumClient;
    uint32_t m_nExtraNonce1;
    uint32_t m_nExtraNonce2;
    EquihashSolverType m_ehSolver;
    EhDevice<EquihashType> m_devStore;
    blake2b_state m_initialState;
	CDataStream m_hdrStream;
    std::vector<typename EquihashType::solution_type> m_vHostSolutions;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_jobStartTime;
    std::chrono::high_resolution_clock::time_point m_jobStopTime;
    std::atomic_uint64_t m_nSolutions;
    std::atomic_uint64_t m_nJobSolutions;

	std::condition_variable m_cvNewJob;
	std::mutex m_mutexNewJob;
	std::atomic_bool m_bNewJob;
	std::atomic_bool m_bStopCurrentJob;

    size_t getHdrStreamReserveSize() const noexcept;
    std::string getTimeStr() const noexcept;

	void SerializeEquihashInput(CDataStream& ss) const;
};

