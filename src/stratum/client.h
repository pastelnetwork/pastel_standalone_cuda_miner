// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <string>
#include <thread>

#include <src/utils/json-rpc-client.h>
#include <src/equihash/block.h>
#include <src/equihash/equihash.h>

class CStratumClient
{
public:
    CStratumClient(const std::string& sServerAddress, unsigned short nPort);
    virtual ~CStratumClient();

    void handlingLoop();
    void setAuthInfo(const std::string& sLogin, const std::string& sPassword);
    bool submitSolution(const uint32_t nExtraNonce2, const std::string& sTime, const std::string &sHexSolution);
    const uint256 generateNonce(const uint32_t nExtraNonce2) noexcept;

    v_uint8 getEquihashInput() const noexcept;
    uint32_t getExtraNonce1() const noexcept { return m_nExtraNonce1; }
    std::string getPersString() const noexcept { return m_sPersString; }
    uint32_t getN() const noexcept { return N; }
    uint32_t getK() const noexcept { return K; }
    uint256 getNonce() const noexcept { return m_blockHeader.nNonce; }
    uint32_t getTime() const noexcept { return m_blockHeader.nTime; }

protected:
    JsonRpcClient m_JsonRpcClient;

    int m_nRequestId;
    std::string m_sSessionId;
    std::string m_sJobId;
    std::string m_sServerAddress;
    unsigned short m_nPort;
    std::string m_sWorkerName;
    std::string m_sWorkerPassword;
    CEquihashInput m_blockHeader;
    bool m_bCleanJobs;
    uint32_t N;
    uint32_t K;
    double m_difficulty;
    bool m_bConnected;
    std::string m_sPersString;
    uint32_t m_nExtraNonce1;
    std::string m_sTarget;

    void handleNotify(const JsonRpcNotify& notify);
    void handleMiningNotify(const JsonRpcNotify& notify);
    bool connect();
    void disconnect();
    bool authorize();
    bool subscribe();
    bool getDifficulty();
    bool setDifficulty(double difficulty);
    bool reconnect();
    static void mining_job(CStratumClient* client);
};
