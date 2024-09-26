// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <compat/byteswap.h>
#include <src/stratum/miner.h>
#include <src/utils/random.h>
#include <src/utils/strencodings.h>
#include <src/utils/streams.h>
#include <src/utils/hash.h>
#include <src/utils/logger.h>
#include <src/utils/logger.h>
#include <src/utils/uint256.h>
#include <src/equihash/equihash.h>
#include <src/equihash/equihash-helper.h>
#include <src/equihash/equihash-check.h>
#include <src/equihash/blake2b_host.h>
#include <src/stratum/client.h>
#include <local_types.h>
#include <src/kernel/memutils.h>
#include <src/kernel/kernel.h>

using namespace std;
using namespace spdlog;
using namespace chrono;
using namespace std::chrono_literals;

constexpr auto POOL_NO_JOBS_RECONNECT_SECS = 60s;

CMiningThread::CMiningThread(IStratumClient *pStratumClient) : 
    CStoppableServiceThread("MiningThread"),
	m_pStratumClient(pStratumClient),
    m_nExtraNonce1(0),
    m_nExtraNonce2(0),
	m_ehSolver(),
	m_devStore(true),
    m_hdrStream(SER_NETWORK, PROTOCOL_VERSION),
    m_nSolutions(0),
    m_nJobSolutions(0)
{
    m_vHostSolutions.reserve(EhDevice<EquihashType>::MaxSolutions);
    m_hdrStream.reserve(getHdrStreamReserveSize());
    m_startTime = high_resolution_clock::now();
}

size_t CMiningThread::getHdrStreamReserveSize() const noexcept
{
    // header + nonce + solution
	return m_blockHeader.GetReserveSize() + uint256::SIZE + Eh200_9::SolutionWidth + 3;
}

void CMiningThread::SerializeEquihashInput(CDataStream& ss) const
{
    ss.clear();
	ss << m_blockHeader;
}

void CMiningThread::NewJobNotify()
{
	lock_guard<mutex> lock(m_mutexNewJob);
	m_bNewJob = true;
	m_cvNewJob.notify_one();
}

void CMiningThread::StopCurrentJob()
{
	m_bStopCurrentJob = true;
	m_devStore.bBreakSolver = true;
}

void CMiningThread::AssignNewJob()
{
    m_jobStopTime = high_resolution_clock::time_point::min();
	m_devStore.bBreakSolver = true;

    // initialize extra nonce 2 with random value
    m_nExtraNonce2 = random_uint32();

    // Initialize and update the first blake2b_state
    m_ehSolver.InitializeState(m_initialState, m_pStratumClient->getPersString());
    SerializeEquihashInput(m_hdrStream);

    v_uint8 vEquihashInput;
    m_hdrStream.extractData(vEquihashInput);
    blake2b_update_host(&m_initialState, vEquihashInput.data(), vEquihashInput.size());
}

void CMiningThread::execute()
{
    while (!shouldStop())
    {
		bool bAssignedNewJob = m_bNewJob;
        if (m_bNewJob)
		{
			AssignNewJob();
			m_bNewJob = false;
		}

		if (bAssignedNewJob)
            miningLoop(numeric_limits<uint32_t>::max());

    	// wait on cv for new job
		unique_lock<mutex> lock(m_mutexNewJob);
        if (!m_cvNewJob.wait_for(lock, POOL_NO_JOBS_RECONNECT_SECS, [this] { return m_bNewJob || shouldStop(); }))
        {
			// check if last job was stopped more than POOL_NO_JOBS_RECONNECT_SECS ago
			if (duration_cast<seconds>(high_resolution_clock::now() - m_jobStopTime) > POOL_NO_JOBS_RECONNECT_SECS)
			{
				gl_console_logger->info("No new jobs for {} seconds, reconnecting to the pool", POOL_NO_JOBS_RECONNECT_SECS.count());
				m_pStratumClient->reconnect();
			}
        }
    }
}

uint256 CMiningThread::generateNonce(const uint32_t nExtraNonce2) noexcept
{
    m_blockHeader.nNonce.SetNull();
    m_blockHeader.nNonce.SetUint32(0, m_nExtraNonce1);
    m_blockHeader.nNonce.SetUint32(1, nExtraNonce2);
    return m_blockHeader.nNonce;
}

string CMiningThread::getTimeStr() const noexcept
{
    uint32_t nReversedTime = bswap_32(static_cast<uint32_t>(m_blockHeader.nTime));

    stringstream ss;
	ss << hex << setw(8) << setfill('0') << nReversedTime << dec;
	return ss.str();
}

bool CMiningThread::submitSolution(const string& sNonce2, const string& sTime, const string &sHexSolution)
{
    try
	{
		return m_pStratumClient->submitSolution(sNonce2, sTime, sHexSolution);
    } catch (const JsonRpcException& e)
    {
        if ((e.Code() == RPC_ERROR_CODE::POOL_ERROR) && (e.getPoolErrorCode() == 21))
		{
			// job not found
            StopCurrentJob();
		}
        return false;
    }
}

void GetCompactSize(v_uint8 &v, const uint64_t nSize)
{
    if (nSize < 253)
    {
        // Values less than 253 are stored as a single byte.
        v.push_back(static_cast<uint8_t>(nSize));
    }
    else if (nSize <= 0xFFFFu)
    {
        // Values between 253 and 65535 are stored as 0xFD followed by the number as a little-endian uint16_t.
        v.push_back(253);
        const uint16_t data = htole16(static_cast<uint16_t>(nSize));
        v.push_back(static_cast<uint8_t>(data & 0xFF)); // Lower byte
        v.push_back(static_cast<uint8_t>((data >> 8) & 0xFF)); // Higher byte
    }
    else if (nSize <= 0xFFFFFFFFu)
    {
        // Values between 65536 and 4294967295 are stored as 0xFE followed by the number as a little-endian uint32_t.
        v.push_back(254);
        const uint32_t data = htole32(static_cast<uint32_t>(nSize));
        for (int i = 0; i < 4; ++i) // 4 bytes for uint32_t
        {
            v.push_back(static_cast<uint8_t>((data >> (i * 8)) & 0xFF));
        }
    }
    else
    {
        // Values larger than 4294967295 are stored as 0xFF followed by the number as a little-endian uint64_t.
        v.push_back(255);
        const uint64_t data = htole64(nSize);
        for (int i = 0; i < 8; ++i) // 8 bytes for uint64_t
        {
            v.push_back(static_cast<uint8_t>((data >> (i * 8)) & 0xFF));
        }
    }
}

uint32_t CMiningThread::miningLoop(const size_t nIterations)
{
    uint32_t nTotalSolutionCount = 0;

	string sJobId = strprintf("[\x1b[33m%s\x1b[0m] ", m_pStratumClient->getJobId());
    string sSolutionInfo;
    bool bNewJobDetected = false;
	const auto fnCheckNewJob = [this, &sJobId, &bNewJobDetected]() -> bool
    {
        if (shouldStop())
        {
            if (!bNewJobDetected)
            {
                gl_console_logger->debug("{} exiting", sJobId);
                bNewJobDetected = true;
            }
			return true;
        }
        if (m_bNewJob)
        {
            if (!bNewJobDetected)
            {
                gl_console_logger->debug("{} new job detected ==> [{}]", sJobId, m_pStratumClient->getNewJobId());
                bNewJobDetected = true;
            }
            return true;
        }
		if (m_bStopCurrentJob)
		{
			gl_console_logger->info("{} stopping current job", sJobId);
			m_bStopCurrentJob = false;
			return true;
		}
        return false;
    };
    gl_console_logger->info("{} new job started, solps={:.2f}, total solps={:.2f}, solutions: {}", 
        sJobId, getJobSolPs(), getTotalSolPs(), m_nSolutions.load());
    m_jobStartTime = high_resolution_clock::now();
    m_nJobSolutions = 0;
    for (uint32_t i = 0; i < nIterations; ++i)
    {
        if (fnCheckNewJob())
            break;

        blake2b_state currState = m_initialState;
        const uint256 nonce = generateNonce(m_nExtraNonce2);
        blake2b_update_host(&currState, nonce.begin(), nonce.size());

        // Copy blake2b states from host to the device
        copyToDevice(m_devStore.d_initialState.get(), &currState, sizeof(currState));

        uint256 nonceReversed = nonce;
        nonceReversed.Reverse();
		gl_console_logger->debug("{} solving for nonce << {} >>", sJobId, nonceReversed.ToString());
        const uint32_t nSolutionCount = m_devStore.solve();
        
        nTotalSolutionCount += nSolutionCount;
        if (nSolutionCount == 0)
        {
			gl_console_logger->debug("{} no solutions found for extra nonce 2: {}", sJobId, m_nExtraNonce2);
            ++m_nExtraNonce2;
            continue;
        }
        //gl_console_logger->info("{} solutions", nSolutionCount);
        m_nJobSolutions += nSolutionCount;
        m_nSolutions += nSolutionCount;

        m_devStore.copySolutionsToHost(m_vHostSolutions);

        DBG_EQUI_WRITE_FN(m_devStore.debugWriteSolutions(m_vHostSolutions));

        // check solutions
        v_uint8 solutionMinimal;
        v_uint32 vSolution;
        vSolution.resize(EquihashType::ProofSize);
        string sError;
        size_t num = 1;

		// get reversed little-endian time in hex when miner found the solution
        // for now just use the time provided by the pool
		string sTime = getTimeStr();

		// nonce is 32 bytes, 64 bytes in hex
		// return part of the nonce without first extraNonce1 bytes
        // need to reverse extraNonce2
		stringstream ss;
		ss << hex << setw(8) << setfill('0') << bswap_32(m_nExtraNonce2);
		// fill the rest of the nonce with zeros
		ss << setw(48) << setfill('0') << 0 << dec;
		string sNonce2 = ss.str();

        // generate solution width in compact form in hex
        v_uint8 vSolutionCompactSize;
        GetCompactSize(vSolutionCompactSize, EquihashType::SolutionWidth);

		const size_t nSavedHdrStreamSize = m_hdrStream.size();

        string sHexSolutionSize = HexStr(vSolutionCompactSize);
        string sHexSolution;
        sHexSolution.reserve(2 * (GetSizeOfCompactSize(EquihashType::SolutionWidth) + EquihashType::SolutionWidth));

        for (const auto& solution : m_vHostSolutions)
        {
            memcpy(vSolution.data(), &solution.indices, sizeof(solution.indices));
            // string s;
            // for (auto i : vSolution)
            //     s += to_string(i) + " ";
            // cout << "new solution indices:" << s << endl;

            solutionMinimal = GetMinimalFromIndices(vSolution, EquihashType::CollisionBitLength);
            /*
            if (!m_ehSolver.IsValidSolution(sError, currState, solutionMinimal))
            {
				gl_console_logger->error("{} invalid solution: {}", sJobId, sError);
                ++num;
                continue;
            }
            */
			m_hdrStream << nonce;
			m_hdrStream << solutionMinimal;
			//cout << sJobId << "Header: " << HexStr(m_hdrStream.cbegin(), m_hdrStream.cend()) << endl;
			uint256 hdrHash = Hash(m_hdrStream.cbegin(), m_hdrStream.cend());

            sSolutionInfo = strprintf("#%zu-%u", num, solution.mainIndex);
			gl_console_logger->debug("{} {:<9}: {}", sJobId, sSolutionInfo, hdrHash.ToString());
			m_hdrStream.resize(nSavedHdrStreamSize);

            if (CheckMinerSolution(hdrHash, m_pStratumClient->getTarget()))
            {
                sHexSolution = sHexSolutionSize + HexStr(solutionMinimal);
                if (submitSolution(sNonce2, sTime, sHexSolution))
            		gl_console_logger->info("{}\x1b[32msolution accepted by the pool [{}]\x1b[0m", sJobId, hdrHash.ToString());
                else
		            gl_console_logger->error("{}\x1b[31msolution rejected by the pool [{}]\x1b[0m", sJobId, hdrHash.ToString());
            }
            ++num;

            if (fnCheckNewJob())
                break;
        }

        ++m_nExtraNonce2;
    }

    m_jobStopTime = high_resolution_clock::now();
    return nTotalSolutionCount;
}

double CMiningThread::getJobSolPs() const noexcept
{
    const auto currentTime = high_resolution_clock::now();
    const auto nJobTimeSecs = duration_cast<seconds>(currentTime - m_jobStartTime).count();
    return nJobTimeSecs ? static_cast<double>(m_nJobSolutions) / nJobTimeSecs : 0.0;
}

double CMiningThread::getTotalSolPs() const noexcept
{
    const auto currentTime = high_resolution_clock::now();
    const auto nTimeSecs = duration_cast<seconds>(currentTime - m_startTime).count();
    return nTimeSecs ? static_cast<double>(m_nSolutions) / nTimeSecs : 0.0;
}
