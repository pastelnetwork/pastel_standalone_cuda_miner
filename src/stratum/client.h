#pragma once
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

#include <string>
#include <thread>

#include <src/utils/arith_uint256.h>
#include <src/equihash/equihash.h>
#include <src/stratum/json-rpc-client.h>
#include <src/stratum/client_intf.h>
#include <src/stratum/miner.h>

class CStratumClient : public IStratumClient
{
public:
    CStratumClient();
    virtual ~CStratumClient();

	virtual bool initMiningThread(std::string &error);
    void setServerInfo(const std::string& sServerAddress, unsigned short nServerPort);
    void startHandlingLoop();
    void breakHandlingLoop();
    void setAuthInfo(const size_t nMinerId, const std::string& sLogin, const std::string& sPassword);
    void checkConnectionState();
    void disconnect();
    bool submitSolution(const std::string& sNonce2, const std::string& sTime, const std::string& sHexSolution) override;
    bool reconnect() override;
	void AssignNewJob() override;

    std::string getPersString() const noexcept override { return m_sPersString; }
	std::string getJobId() const noexcept override { return m_sJobId; }
	arith_uint256 getTarget() const noexcept override { return m_target; }
	std::string getNewJobId() const noexcept override { return m_sNewJobId; }

    uint32_t getN() const noexcept { return N; }
    uint32_t getK() const noexcept { return K; }
    
protected:
    JsonRpcClient m_JsonRpcClient;
	std::unique_ptr<CMiningThread> m_pMiningThread;

	size_t m_nMinerId;
    int m_nRequestId;
    std::string m_sSessionId;
    std::string m_sJobId;
    std::string m_sServerAddress;
    unsigned short m_nServerPort;
    std::string m_sWorkerName;
    std::string m_sWorkerPassword;
    bool m_bCleanJobs;
    uint32_t N;
    uint32_t K;
    double m_difficulty;
    bool m_bConnected;
    std::string m_sPersString;
	arith_uint256 m_target;

    CEquihashInput m_NewJobBlockHeader;
	std::string m_sNewJobId;
	uint256 m_NewTarget;

    void handleNotify(const JsonRpcNotify& notify);
    void handleMiningNotify(const JsonRpcNotify& notify);
    void handleSetTargetNotify(const JsonRpcNotify& notify);
    bool connect();
    bool authorize();
    bool subscribe();
    bool getDifficulty();
    bool setDifficulty(double difficulty);
};
