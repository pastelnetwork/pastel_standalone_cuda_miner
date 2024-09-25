#pragma once
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <string>

#include <src/utils/arith_uint256.h>

class IStratumClient
{
public:
	virtual ~IStratumClient() = default;

    virtual std::string getPersString() const noexcept = 0;
	virtual std::string getJobId() const noexcept = 0;
	virtual arith_uint256 getTarget() const noexcept = 0;
	virtual std::string getNewJobId() const noexcept = 0;

    virtual bool submitSolution(const std::string& sNonce2, const std::string& sTime, const std::string& sHexSolution) = 0;
	virtual bool reconnect() = 0;
	virtual void AssignNewJob() = 0;
};
