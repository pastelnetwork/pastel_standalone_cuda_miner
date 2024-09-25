// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <string>
#include <functional>
#include <optional>
#include <variant>
#include <map>
#include <vector>
#include <unordered_set>
#include <exception>
#include <condition_variable>
#include <mutex>

#include <compat.h>

#include <event2/event.h>
#include <event2/bufferevent.h>
#include <nlohmann/json.hpp>

#include <rpc_error.h>
#include <src/utils/svc_thread.h>

bool json_rpc_has_key(const nlohmann::json &v, const std::string &key);
bool json_rpc_has_key_type(const nlohmann::json &v, const std::string &key, nlohmann::json::value_t type);
bool json_rpc_valid_id(const nlohmann::json &request);
bool json_rpc_valid_id_not_null(const nlohmann::json &request);

class JsonRpcException : public std::exception
{
public:
    JsonRpcException(RPC_ERROR_CODE code, const std::string &message) noexcept;
	JsonRpcException(const std::string& message, const int nErrorCode) noexcept;
    JsonRpcException(RPC_ERROR_CODE code, const std::string &message, const nlohmann::json &data) noexcept;

    RPC_ERROR_CODE Code() const noexcept { return code; }
	int getPoolErrorCode() const noexcept { return nCode; }
    const std::string &Message() const noexcept { return message; }
    const nlohmann::json &Data() const noexcept { return data; }

    const char *what() const noexcept override { return err.c_str(); }

    static inline JsonRpcException fromJson(const nlohmann::json &value)
    {
        const bool bHasCode = json_rpc_has_key_type(value, "code", nlohmann::json::value_t::number_integer);
        const bool bHasMessage = json_rpc_has_key_type(value, "message", nlohmann::json::value_t::string);
        bool bHasData = json_rpc_has_key(value, "data");
        if (bHasCode && bHasMessage)
        {
            int nCode = value["code"].get<int>();
            RPC_ERROR_CODE code = RPC_ERROR_CODE::INTERNAL;
            if (is_rpc_error_code_valid(nCode))
                code = static_cast<RPC_ERROR_CODE>(nCode);
            if (bHasData)
                return JsonRpcException(code, value["message"], value["data"].get<nlohmann::json>());

            return JsonRpcException(code, value["message"]);
        }
        return JsonRpcException(RPC_ERROR_CODE::INTERNAL, R"(invalid error response: "code" (negative number) and "message" (string) are required)");
    }

private:
    RPC_ERROR_CODE code;
	int nCode;
    std::string message;
    nlohmann::json data;
    std::string err;
};

typedef std::variant<std::monostate, int, std::string> id_type;

std::string to_string(const id_type& id);

struct JsonRpcResponse
{
    id_type id;
    nlohmann::json result;
};

struct JsonRpcRequest
{
    id_type id;
    std::string method;
    nlohmann::json params;
};

struct JsonRpcNotify : JsonRpcRequest
{
    bool isResult = false;
    std::string error;
};

class JsonRpcClient : public CStoppableServiceThread
{
public:
    enum class rpcVersion { v1, v2 };
    typedef std::vector<nlohmann::json> positional_parameter;
    typedef std::map<std::string, nlohmann::json> named_parameter;

    enum class ConnectionState
    {
        SRV_DISCONNECTED,
        SRV_CONNECTED,
        SRV_TIMEOUT,
        SRV_ERROR,
        CLOSED_BY_SERVER
    };
    using ResponseCallback = std::function<void(const nlohmann::json& response)>;
    using ErrorCallback = std::function<void(const std::string& error)>;

    JsonRpcClient(const char *szThreadName);
    ~JsonRpcClient();

    void setServerInfo(const std::string& sServerAddress, unsigned short nServerPort);
    bool Connect();
    void Disconnect();
    ConnectionState GetConnectionState() const noexcept { return m_connectionState; }

    void SetResponseCallback(const ResponseCallback &callback) noexcept { m_responseCallback = callback; }
    void SetErrorCallback(const ErrorCallback &callback) noexcept { m_errorCallback = callback; }
    void SetConnectionState(const ConnectionState state, const char *szError = nullptr) noexcept;

    void BreakEventLoop();

    template <typename T>
    T CallMethod(const id_type &id, const std::string &name)
    {
        return call_method(id, name, nlohmann::json::object()).result.get<T>();
    }

     template <typename T>
    T CallMethod(const id_type &id, const std::string &name, const positional_parameter &params)
    {
        return call_method(id, name, params).result.get<T>();
    }
    template <typename T>
    T CallMethodNamed(const id_type &id, const std::string &name, const named_parameter &params = {})
    {
        return call_method(id, name, params).result.get<T>();
    }
    static JsonRpcNotify ParseJsonRpcNotify(const nlohmann::json &j);
    bool addSyncRequestId(const id_type requestId);
    bool removeSyncRequestId(const id_type requestId);
    bool handleReceivedId(const id_type &requestId);

	void execute() override;

protected:
    void FreeConnection();
    bool Send(std::string &error, const std::string& request);
    bool SendAndReceive(std::string &error, const std::string& request, const id_type requestId);

private:
    rpcVersion m_rpcVersion;
    std::string m_sServerAddress;
    unsigned short m_nServerPort;
    struct event_base* m_pEventBase;
    struct bufferevent* m_bufferEvent;
    std::string m_sResponseBuffer;
    std::optional<nlohmann::json> m_responseJson;
    std::string m_sError;
    ResponseCallback m_responseCallback;
    ErrorCallback m_errorCallback;
    ConnectionState m_connectionState;
    bool m_bSyncRequestPending;
	std::condition_variable m_cvSyncRequest;
	std::mutex m_SyncRequestMutex;
    std::unordered_set<id_type> m_IdSent;
	std::unordered_set<id_type> m_IdReceived;
	std::mutex m_IdMutex;
	bool m_bBreakEventLoop;

    static void OnEventCallback(struct bufferevent* bev, short events, void* pContext);
    static void bev_callback(struct bufferevent* bev, void* pContext);
    static std::optional<nlohmann::json> ParseJsonObject(const std::string& sJson, std::string &error);

    JsonRpcResponse call_method(const id_type &id, const std::string &name, const nlohmann::json &params);
};
