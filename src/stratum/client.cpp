// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <chrono>
#include <thread>

#include <scope_guard.hpp>

#include <src/utils/str_utils.h>
#include <src/utils/strencodings.h>
#include <src/utils/streams.h>
#include <src/utils/serialize.h>
#include <src/equihash/block.h>
#include <src/stratum/client.h>
#include <src/stratum/miner.h>

using namespace std;
using namespace std::chrono_literals;

using json = nlohmann::json;

constexpr auto POOL_RECONNECT_INTERVAL_SECS = 15s;

unique_ptr<CMiningThread> gl_MiningThread;

CStratumClient::CStratumClient(const std::string& sServerAddress, unsigned short nPort) :
    m_sServerAddress(sServerAddress),
    m_nPort(nPort),
    m_nRequestId(0),
    m_bCleanJobs(false),
    N(200),
    K(9),
    m_difficulty(0),
    m_bConnected(false)
{}    

CStratumClient::~CStratumClient()
{
    disconnect();
}

bool CStratumClient::connect()
{
    cout << "Connecting to [" << m_sServerAddress << ":" << m_nPort << "]" << endl;
    if (!m_JsonRpcClient.Connect(m_sServerAddress, m_nPort))
        return false;
    cout << "...connected" << endl;
    return true;
}

void CStratumClient::disconnect()
{
    m_JsonRpcClient.Disconnect();
    m_bConnected = false;
}

void  CStratumClient::setAuthInfo(const string& sWorkerName, const string& sWorkerPassword)
{
    m_sWorkerName = sWorkerName;
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
    cout << "Authorizing with the pool..."  << endl;
    bool bResult = m_JsonRpcClient.CallMethod<bool>(
        ++m_nRequestId,
        "mining.authorize", 
        { m_sWorkerName, m_sWorkerPassword });
    if (bResult)
        cout << "...successfully authorized" << endl;
    else
        cerr << "Failed to authorize with the pool" << endl;
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
    cout << "Subscribing to the pool..." << endl;
    auto result = m_JsonRpcClient.CallMethod<json::array_t>(
        ++m_nRequestId,
        "mining.subscribe");
    if (result.size() > 0)
    {
        string sExtraNonce1 = result[1].get<std::string>();
        // convert hex string to uint32
        m_nExtraNonce1 = stoul(sExtraNonce1, nullptr, 16);
        cout << "...successfully subscribed, extraNonce1=[0x" << sExtraNonce1 << "]" << endl;
        return true;
    }
    cerr << "Failed to subscribe to the pool" << endl;
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
        cerr << "Failed to authorize with the pool" << endl;
        return false;
    }

    if (!subscribe())
    {
        cerr << "Failed to subscribe to the pool" << endl;
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
bool CStratumClient::submitSolution(const uint32_t nExtraNonce2, const string& sTime, const string &sHexSolution)
{
    if (!m_bConnected)
        return false;

    auto params = json::array({m_sWorkerName, m_sJobId, sTime, HexStr(nExtraNonce2), sHexSolution});
    bool bResult = m_JsonRpcClient.CallMethod<bool>(++m_nRequestId, "mining.submit", params);
    if (bResult)
        cout << "Solution accepted by the pool" << endl;
    else
        cerr << "Solution rejected by the pool" << endl;
    return bResult;
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

void CStratumClient::handleMiningNotify(const JsonRpcNotify &notify)
{
    const auto &params = notify.params;
    if (params.is_null() || !params.is_array())
    {
        return;
    }
    if (params.size() < 12)
    {
        cerr << notify.method << ": invalid number of parameters (less than 12)" << endl;
        return;
    }
    const auto fnValidateHashParam = [&](const char *szHashDesc, const json& v, uint256 &hash) -> bool
    {
        if (!v.is_string())
        {
            cerr << notify.method << ": invalid " << szHashDesc << " hash value" << endl;
            return false;
        }
        string error;
        string sHashValue = v.get<std::string>();
        if (!parse_uint256(error, hash, sHashValue, szHashDesc))
        {
            cerr << notify.method << ": invalid " << szHashDesc << " hash value, " << error << endl;
            return false;
        }
        return true;
    };
    const auto &v = params[0];
    if (!v.is_string())
    {
        cerr << notify.method << ": invalid job ID" << endl;
        return;
    }
    m_sJobId = v.get<std::string>();
    m_blockHeader.Clear();
    // block version parameter uint32_t LE
    m_blockHeader.nVersion = stoul(params[1].get<std::string>(), nullptr, 16);

    if (!fnValidateHashParam("previous block", params[2], m_blockHeader.hashPrevBlock) ||
        !fnValidateHashParam("merkle root", params[3], m_blockHeader.hashMerkleRoot) ||
        !fnValidateHashParam("final sapling root", params[4], m_blockHeader.hashFinalSaplingRoot))
    {
        return;
    }
    // nTime uint32 LE
    m_blockHeader.nTime = stoul(params[5].get<std::string>(), nullptr, 16);
    // nBits uint32 LE
    m_blockHeader.nBits = stoul(params[6].get<std::string>(), nullptr, 16);
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
            cerr << notify.method << ": invalid algoNK value" << endl;
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
    m_blockHeader.sPastelID = params[10].get<std::string>();
    // previous merkle root signature - hex-encoded string
    string sPrevMerkleRootSignature = params[11].get<std::string>();
    if (!IsHex(sPrevMerkleRootSignature))
    {
        cerr << notify.method << ": invalid previous merkle root signature" << endl;
        return;
    }
    m_blockHeader.prevMerkleRootSignature = ParseHex(sPrevMerkleRootSignature);
    cout << "New job received: " << m_sJobId << endl;

    // start new mining job in a separate thread
    if (gl_MiningThread && gl_MiningThread->isRunning())
        gl_MiningThread->waitForStop();
    gl_MiningThread = make_unique<CMiningThread>(*this);

    string error;
    if (!gl_MiningThread->start(error))
    {
        cerr << "Failed to start mining thread: " << error << endl;
    }
}

v_uint8 CStratumClient::getEquihashInput() const noexcept
{
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    // calculate equihash input
    ss.reserve(m_blockHeader.GetReserveSize());
    ss << m_blockHeader;
    v_uint8 v;
    ss.extractData(v);
    return v;
}

const uint256 CStratumClient::generateNonce(const uint32_t nExtraNonce2) noexcept
{
    m_blockHeader.nNonce.SetNull();
    m_blockHeader.nNonce.SetUint32(0, m_nExtraNonce1);
    m_blockHeader.nNonce.SetUint32(1, nExtraNonce2);
    return m_blockHeader.nNonce;
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
            cout << "New difficulty: " << m_difficulty << endl;
        }
    }
    else if (notify.method == "mining.set_target")
    {
        if (params.is_array() && params.size() >= 1)
        {
            m_sTarget = params[0].get<std::string>();
            cout << "New target: " << m_sTarget << endl;
        }
    }
    else
    {
        cerr << "Unknown notification from stratum pool: " << notify.method << endl;    
    }
}

void CStratumClient::handlingLoop()
{
    m_JsonRpcClient.SetResponseCallback([&](const json& j)
    {
        try
        {
            const JsonRpcNotify jsonRpcNotfy = JsonRpcClient::ParseJsonRpcNotify(j);
            handleNotify(jsonRpcNotfy);
        }
        catch(const JsonRpcException& e)
        {
            cerr << e.what() << endl;
        }
    });        

    while (true)
    {
        if (!m_bConnected && !reconnect())
        {
            this_thread::sleep_for(POOL_RECONNECT_INTERVAL_SECS);
            continue;
        }

        m_JsonRpcClient.EnterEventLoop();
      }
}
