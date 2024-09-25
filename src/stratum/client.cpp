// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <chrono>
#include <thread>

#include <scope_guard.hpp>

#include <src/utils/str_utils.h>
#include <src/utils/strencodings.h>
#include <src/utils/logger.h>
#include <src/equihash/block.h>
#include <src/equihash/equihash.h>
#include <src/stratum/client.h>

using namespace std;
using namespace std::chrono_literals;

using json = nlohmann::json;

CStratumClient::CStratumClient() :
    m_JsonRpcClient("stm-client"),
    m_nServerPort(0),
    m_nRequestId(0),
    m_bCleanJobs(false),
    N(200),
    K(9),
    m_difficulty(0),
    m_bConnected(false),
	m_nMinerId(0)
{}    

CStratumClient::~CStratumClient()
{
    if (m_pMiningThread)
		m_pMiningThread->waitForStop();
    disconnect();
}

bool CStratumClient::initMiningThread(string &error)
{
    // start miner in a separate thread
    m_pMiningThread = make_unique<CMiningThread>(this);
    if (!m_pMiningThread->start(error))
    {
		string excError = "Failed to start mining thread: " + error;
        gl_console_logger->error(excError);
        return false;
    }
	return true;
}

void CStratumClient::setServerInfo(const string& sServerAddress, unsigned short nServerPort)
{
    m_sServerAddress = sServerAddress;
    m_nServerPort = nServerPort;
	m_JsonRpcClient.setServerInfo(sServerAddress, nServerPort);
}

bool CStratumClient::connect()
{
	gl_console_logger->info("Connecting to [{}:{}]", m_sServerAddress, m_nServerPort);
    if (!m_JsonRpcClient.Connect())
        return false;
	gl_console_logger->info("...connected");
    return true;
}

void CStratumClient::disconnect()
{
    m_JsonRpcClient.Disconnect();
    m_bConnected = false;
}

void  CStratumClient::setAuthInfo(const size_t nMinerId, const string& sWorkerName, const string& sWorkerPassword)
{
	m_nMinerId = nMinerId;
    m_sWorkerName = strprintf("%s_%d", sWorkerName, m_nMinerId);
    m_sWorkerPassword = sWorkerPassword;
}

/**
 * @brief Authorizes worker with the pool.
 * 
 * Example:
 *   request:
 *      {"params": ["worker_name.ip", "password"], "id": 2, "method": "mining.authorize"}\n
 *   response:
 *      {"error": null, "id": 2, "result": true}\n
 * @return true 
 * @return false 
 */
bool CStratumClient::authorize()
{
	if (m_sWorkerName.empty() || m_sWorkerPassword.empty())
	{
		gl_console_logger->error("Worker name or password is not set");
		return false;
	}
	gl_console_logger->info("{} authorizing with the pool...", m_sWorkerName);
	bool bResult = false;
    try
    {
        bResult = m_JsonRpcClient.CallMethod<bool>(
            ++m_nRequestId,
            "mining.authorize",
            { m_sWorkerName, m_sWorkerPassword });
    } catch (const JsonRpcException& e)
    {
        gl_console_logger->error("JsonRpcException: {}", e.what());
    }
    if (bResult)
		gl_console_logger->info("...successfully authorized {}", m_sWorkerName);
    else
		gl_console_logger->error("Failed to authorize with the pool worker {}", m_sWorkerName);
    return bResult;
}

/**
 * @brief Subscribes to the pool.
 * This should return subcription details, extranonce1 and extranonce2_size.
 * 
 * example:
 *    request:
 *      {"id": 1, "method": "mining.subscribe", "params": []}\n
 *    response:
 *      {"id": 1, 
 *      "result": [ [ ["mining.set_difficulty", "b4b6693b72a50c7116db18d6497cac52"],
 *                    ["mining.notify", "ae6812eb4cd7735a302a8a9dd95cf71f"] ],
 *                  "08000002", 4],
 *      "error": null}\n
 * parameters are: details, extranonce1, extranonce2_size (optional)
 * @return true if successful, false otherwise
 */
bool CStratumClient::subscribe()
{
    gl_console_logger->info("Subscribing to the pool...");
    try
    {
        auto result = m_JsonRpcClient.CallMethod<json::array_t>(
            ++m_nRequestId,
            "mining.subscribe");
        if (result.size() > 0)
        {
            string sExtraNonce1 = result[1].get<std::string>();
            // convert hex string to uint32
            const uint32_t nExtraNonce1 = ConvertHexToUint32LE(sExtraNonce1);
            gl_console_logger->info("...successfully subscribed, extraNonce1=[0x{:x}]", nExtraNonce1);
            m_pMiningThread->setExtraNonce1(nExtraNonce1);
            return true;
        }
    } catch (const JsonRpcException& e) {
        gl_console_logger->error("Failed to subscribe to the pool. {}", e.what());
    }
    return false;
}

bool CStratumClient::reconnect()
{
    // Close the existing connection, if any
    disconnect();

    // Try to establish a new connection to the pool
    if (!connect())
        return false;

    auto guard = sg::make_scope_guard([&]() noexcept { this->disconnect(); });

    if (!authorize())
    {
		gl_console_logger->error("Failed to authorize with the pool");
        return false;
    }

    if (!subscribe())
    {
		gl_console_logger->error("Failed to subscribe to the pool");
        return false;
    }

    m_bConnected = true;
    guard.dismiss();
    return true;
}

/**
 * Submit a solution to the pool.
 * 
 * mining.submit("workerName", "job id", "ExtraNonce2", "nTime", "nOnce")
 * 
 * Parameters:
 *   [0] workerName - the name of the worker
 *   [1] jobId - the job ID
 *   [2] nTime - the time value
 *   [3] ExtraNonce2 - the extra nonce 2 value
 *   [4] solution
 * 
 * \param nExtraNonce2 - the extra nonce 2 value
 * \param sTime - the time value
 * \param sHexSolution - the solution in hex format
 * \return true if the solution was accepted by the pool, false otherwise
 */
bool CStratumClient::submitSolution(const string& sNonce2, const string& sTime, const string &sHexSolution)
{
    if (!m_bConnected)
        return false;

    auto params = json::array({
        m_sWorkerName,
        m_sJobId,
        sTime,
        sNonce2,
        sHexSolution
    });
    return m_JsonRpcClient.CallMethod<bool>(++m_nRequestId, "mining.submit", params);
}

bool CStratumClient::getDifficulty()
{
    auto result = m_JsonRpcClient.CallMethod<json>(++m_nRequestId, "mining.get_difficulty");
    if (result.is_number_float())
    {
        m_difficulty = result.get<double>();
        return true;
    }
    return false;
}

bool CStratumClient::setDifficulty(double difficulty)
{
    auto params = json::array({difficulty});
    bool bResult = m_JsonRpcClient.CallMethod<bool>(++m_nRequestId, "mining.set_difficulty", params);
    if (bResult)
    {
        m_difficulty = difficulty;
    }
    return bResult;
}

void CStratumClient::handleSetTargetNotify(const JsonRpcNotify& notify)
{
    const auto &params = notify.params;
    if (params.is_null() || !params.is_array() || params.empty())
	{
		gl_console_logger->error("{}: invalid number of parameters", notify.method);
		return;
	}
    string error;
    string sNewTarget = params[0].get<std::string>();
    // validate new target
	if (!parse_uint256(error, m_NewTarget, sNewTarget, "target"))
	{
		gl_console_logger->error("{}: invalid target value, {}", notify.method, error);
		return;
	}
	gl_console_logger->info("New target: {}", sNewTarget);
}

void CStratumClient::handleMiningNotify(const JsonRpcNotify &notify)
{
    const auto &params = notify.params;
    if (params.is_null() || !params.is_array())
    {
        return;
    }
    if (params.size() < 12)
    {
		gl_console_logger->error("{}: invalid number of parameters (less than 12)", notify.method);
        return;
    }
    const auto fnValidateHashParam = [&](const char *szHashDesc, const json& v, uint256 &hash) -> bool
    {
        if (!v.is_string())
        {
			gl_console_logger->error("{}: invalid {} hash value", notify.method, szHashDesc);
            return false;
        }
        string error;
        string sHashValue = v.get<std::string>();
        if (!parse_uint256(error, hash, sHashValue, szHashDesc))
        {
			gl_console_logger->error("{}: invalid {} hash value, {}", notify.method, szHashDesc, error);
            return false;
        }
        hash.Reverse();
        return true;
    };
    const auto &v = params[0];
    if (!v.is_string())
    {
		gl_console_logger->error("{}: invalid job ID", notify.method);
        return;
    }
    m_sNewJobId = v.get<std::string>();
    auto& hdr = m_NewJobBlockHeader;
    hdr.Clear();
    // block version parameter (uint32_t) in big-endian hex format
    hdr.nVersion = ConvertHexToUint32LE(params[1].get<std::string>());

    if (!fnValidateHashParam("previous block", params[2], hdr.hashPrevBlock) ||
        !fnValidateHashParam("merkle root", params[3], hdr.hashMerkleRoot) ||
        !fnValidateHashParam("final sapling root", params[4], hdr.hashFinalSaplingRoot))
    {
        return;
    }
    // nTime uint32
    hdr.nTime = ConvertHexToUint32LE(params[5].get<std::string>());
    // nBits uint32
    hdr.nBits = ConvertHexToUint32LE(params[6].get<std::string>());
    // clean jobs boolean flag
    m_bCleanJobs = params[7].get<bool>();
    
    // algoNK parameter in format N_K, optional - can be null
    const auto &algoNK = params[8];
    if (!algoNK.is_null() && algoNK.is_string())
    {
        string sAlgoNK = algoNK.get<std::string>();
        size_t nPos = sAlgoNK.find('_');
        if (nPos == string::npos)
        {
			gl_console_logger->error("{}: invalid algoNK value", notify.method);
            return;
        }
        N = stoul(sAlgoNK.substr(0, nPos));
        K = stoul(sAlgoNK.substr(nPos + 1));
    }

    // personalization string
    const auto &persString = params[9];
    if (!persString.is_null() && persString.is_string())
        m_sPersString = persString.get<std::string>();
    else
        m_sPersString = DEFAULT_EQUIHASH_PERS_STRING;

    // mnid string parameter
    hdr.sPastelID = params[10].get<std::string>();
    // previous merkle root signature - hex-encoded and base64-encoded string
    string sPrevMerkleRootSignatureEncoded = params[11].get<std::string>();
    if (!IsHex(sPrevMerkleRootSignatureEncoded))
    {
		gl_console_logger->error("{}: invalid previous merkle root signature hex-encoding", notify.method);
        return;
    }
    v_uint8 vMerkleRoot = ParseHex(sPrevMerkleRootSignatureEncoded);
    bool bInvalidBase64Encoding = false;
    hdr.prevMerkleRootSignature = DecodeBase64(vector_to_string(vMerkleRoot).c_str(), &bInvalidBase64Encoding);
    if (bInvalidBase64Encoding)
    { 
		gl_console_logger->error("{}: invalid previous merkle root signature base64-encoding", notify.method);
		return;
	}
	gl_console_logger->info("New job received for miner #{}: [\x1b[33m{}\x1b[0m], target: {}", 
        m_nMinerId, m_sNewJobId, m_NewTarget.ToString());
    gl_console_logger->info("mnid: {}", hdr.sPastelID);

	AssignNewJob();
	m_pMiningThread->NewJobNotify();
}

void CStratumClient::AssignNewJob()
{
	auto& blockHeader = m_pMiningThread->getBlockHeader();
	blockHeader = std::move(m_NewJobBlockHeader);
	m_sJobId = std::move(m_sNewJobId);
	m_target = UintToArith256(m_NewTarget);
}

void CStratumClient::handleNotify(const JsonRpcNotify &notify)
{
    const auto &params = notify.params;
    if (notify.method == "mining.notify")
    {
        handleMiningNotify(notify);
    }
    else if (notify.method == "mining.set_difficulty")
    {
        if (params.is_array() && params.size() >= 1)
        {
            m_difficulty = params[0].get<double>();
			gl_console_logger->info("[\x1b[33m{}\x1b[0m] new difficulty: {}", m_sNewJobId, m_difficulty);
        }
    }
    else if (notify.method == "mining.set_target")
    {
        handleSetTargetNotify(notify);
    }
    else
    {
		gl_console_logger->error("Unknown notification from stratum pool: {}", notify.method);
    }
}

void CStratumClient::startHandlingLoop()
{
    m_JsonRpcClient.SetResponseCallback([&](const json& j)
    {
        try
        {
            const JsonRpcNotify jsonRpcNotify = JsonRpcClient::ParseJsonRpcNotify(j);
			if (!m_JsonRpcClient.handleReceivedId(jsonRpcNotify.id))
                handleNotify(jsonRpcNotify);
        }
        catch(const JsonRpcException& e)
        {
			gl_console_logger->error("JsonRpcException: {}", e.what());
        }
    });        

    string error;
    if (!m_JsonRpcClient.start(error))
	{
		gl_console_logger->error("Failed to start stratum pool client. {}", error);
		return;
	}
}

void CStratumClient::breakHandlingLoop()
{
	disconnect();
	m_JsonRpcClient.BreakEventLoop();
    m_JsonRpcClient.waitForStop();
}

void CStratumClient::checkConnectionState()
{
    const auto state = m_JsonRpcClient.GetConnectionState();
    if (state == JsonRpcClient::ConnectionState::SRV_DISCONNECTED ||
        state == JsonRpcClient::ConnectionState::SRV_ERROR)
		m_bConnected = false;

    if (!m_bConnected)
        reconnect();
}

