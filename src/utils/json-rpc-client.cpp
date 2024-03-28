// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <iostream>
#include <memory>
#include <chrono>

#include <event2/buffer.h>
#include <scope_guard.hpp>
#include <tinyformat.h>

#include <src/utils/json-rpc-client.h>
#include <src/utils/enum_util.h>

using namespace std;
using namespace std::chrono_literals;
using json = nlohmann::json;

constexpr auto JSON_RPC_RESPONSE_TIMEOUT_SECS = 15s;

bool json_rpc_has_key(const json &v, const string &key)
{
    return v.find(key) != v.end();
}

bool json_rpc_has_key_type(const json &v, const string &key, json::value_t type)
{
    return json_rpc_has_key(v, key) && v.at(key).type() == type;
}

bool json_rpc_valid_id(const json &request)
{
    return json_rpc_has_key(request, "id") && (request["id"].is_number() || request["id"].is_string() || request["id"].is_null());
}

bool json_rpc_valid_id_not_null(const json &request)
{
    return json_rpc_has_key(request, "id") && (request["id"].is_number() || request["id"].is_string());
}

JsonRpcException::JsonRpcException(RPC_ERROR_CODE code, const std::string &message) noexcept : 
    code(code),
    message(message),
    data(nullptr),
    err(to_string(to_integral_type(code)) + ": " + message)
{}

JsonRpcException::JsonRpcException(RPC_ERROR_CODE code, const std::string &message, const json &data) noexcept :
    code(code),
    message(message),
    data(data),
    err(to_string(to_integral_type(code)) + ": " + message + ", data: " + data.dump())
{}

JsonRpcClient::JsonRpcClient() :
    m_rpcVersion(rpcVersion::v2),
    m_nPort(0),
    m_eventBase(event_base_new()),
    m_bufferEvent(nullptr),
    m_connectionState(ConnectionState::DISCONNECTED),
    m_bSyncRequestPending(false)
{
    m_sResponseBuffer.reserve(2048);
}

JsonRpcClient::~JsonRpcClient()
{
    FreeConnection();
    if (m_eventBase)
        event_base_free(m_eventBase);
}

void JsonRpcClient::Disconnect()
{
    FreeConnection();
}

void JsonRpcClient::FreeConnection()
{
   if (m_bufferEvent)
   {
        bufferevent_free(m_bufferEvent);
        m_bufferEvent = nullptr;
   }
 }

bool JsonRpcClient::Connect(const string& sServerAddress, unsigned short nPort)
{
    FreeConnection();
    m_sServerAddress = sServerAddress;
    m_nPort = nPort;

    m_bufferEvent = bufferevent_socket_new(m_eventBase, -1, BEV_OPT_CLOSE_ON_FREE);
    if (!m_bufferEvent)
    {
        cerr << "Failed to create buffer event" << endl;
        return false;
    }

    // Set read/write timeouts
    // struct timeval timeout;
    // timeout.tv_sec = JSON_RPC_RESPONSE_TIMEOUT_SECS.count(); // Set timeout to 10 seconds
    // timeout.tv_usec = 0;
    // bufferevent_set_timeouts(m_bufferEvent, &timeout, &timeout);

    if (bufferevent_socket_connect_hostname(m_bufferEvent, nullptr, AF_INET, m_sServerAddress.c_str(), m_nPort) < 0)
    {
        cerr << "Failed to connect to [" << m_sServerAddress << ":" << m_nPort << "]" << endl;
        return false;
    }

    bufferevent_setcb(m_bufferEvent, OnDataReceived, nullptr, OnEventCallback, this);
    bufferevent_enable(m_bufferEvent, EV_READ | EV_WRITE);
    return true;
}

void JsonRpcClient::SetConnectionState(const ConnectionState state, const char *szError) noexcept
{
    m_connectionState = state;
    if (szError)
        m_sError = szError;
    if (m_connectionState == ConnectionState::DISCONNECTED)
    {
        FreeConnection();
    }
}

JsonRpcResponse JsonRpcClient::call_method(const id_type &id, const std::string &name, const json &params)
{
    json j = {{"method", name}};
    if (get_if<int>(&id) != nullptr)
        j["id"] = std::get<int>(id);
    else
        j["id"] = std::get<std::string>(id);

    if (m_rpcVersion == rpcVersion::v2)
        j["jsonrpc"] = "2.0";

    if (!params.empty() && !params.is_null())
        j["params"] = params;
    else if (params.is_array())
        j["params"] = params;
    else if (m_rpcVersion == rpcVersion::v1)
        j["params"] = nullptr;

    try
    {
        string error;
        const bool bSucceeded = Send(error, j.dump());
        if (!bSucceeded)
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, error);

        if (!m_responseJson.has_value())
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, "Json-RPC response is empty");

        const json &response = *m_responseJson;
        if (json_rpc_has_key_type(response, "error", json::value_t::object))
            throw JsonRpcException::fromJson(response["error"]);

        if (json_rpc_has_key_type(response, "error", json::value_t::string))
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, response["error"]);
            
        if (json_rpc_has_key(response, "result") && json_rpc_has_key(response, "id"))
        {
            if (response["id"].type() == json::value_t::string)
                return JsonRpcResponse{response["id"].get<std::string>(), response["result"].get<json>()};

            return JsonRpcResponse{response["id"].get<int>(), response["result"].get<json>()};
        }

        throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, R"(invalid server response: neither "result" nor "error" fields found)");
    } catch (json::parse_error &e) {
        throw JsonRpcException(RPC_ERROR_CODE::PARSE, std::string("invalid JSON response from server: ") + e.what());
    }
}

JsonRpcNotify JsonRpcClient::ParseJsonRpcNotify(const json &j)
{
    JsonRpcNotify notify;

    // Extract 'id' field
    if (!j.contains("id"))
        throw JsonRpcException(RPC_ERROR_CODE::INVALID_REQUEST,
            "JsonRpcRequest must contain an 'id' field.");

    if (j["id"].is_number_integer())
        notify.id = j["id"].get<int>();
    else if (j["id"].is_string())
        notify.id = j["id"].get<std::string>();
    else
        throw JsonRpcException(RPC_ERROR_CODE::INVALID_REQUEST, 
            "JsonRpcRequest 'id' field must be either integer or string.");

    // Extract 'method' field
    if (!j.contains("method") || !j["method"].is_string())
        throw JsonRpcException(RPC_ERROR_CODE::METHOD_NOT_FOUND,
            "JsonRpcRequest must contain a 'method' field of string type.");

    notify.method = j["method"].get<std::string>();

    // Extract 'params' field
    if (j.contains("params"))
        notify.params = j["params"];
    else
        notify.params = json::object();

    return notify;
}

bool JsonRpcClient::Send(string &error, const string &request)
{
    m_sError.clear();
    m_responseJson.reset();
    if (!m_bufferEvent)
    {
        error = "Not connected to the server";
        return false;
    }
    cout << request << endl;
    if (bufferevent_write(m_bufferEvent, request.c_str(), request.size()) < 0 ||
        bufferevent_write(m_bufferEvent, "\n", 1) < 0)
    {
        error = "Failed to send request";
        if (!m_sError.empty())
            error += ": " + m_sError;
        return false;
    }

    // Wait for the full json-rpc response, it will be processed in OnDataReceived 
    // and saved in m_responseJson
    m_bSyncRequestPending = true;
    {
        auto guard = sg::make_scope_guard([&]() noexcept { m_bSyncRequestPending = false; });
        EnterEventLoop();
        cout << "Response received" << endl;
    }

    return true;
}

void JsonRpcClient::OnEventCallback(struct bufferevent* bev, short events, void* pContext)
{
    auto* pClient = static_cast<JsonRpcClient*>(pContext);
    if (events & BEV_EVENT_CONNECTED)
    {
        pClient->SetConnectionState(ConnectionState::CONNECTED);
    }
    else if (events & (BEV_EVENT_ERROR | BEV_EVENT_EOF | BEV_EVENT_TIMEOUT))
    {
        if (events & BEV_EVENT_EOF)
        {
            cerr << "Connection closed by the server" << endl;
        } else if (events & BEV_EVENT_ERROR)
        {
            int err = EVUTIL_SOCKET_ERROR();
            cerr << "Socket error: " << evutil_socket_error_to_string(err) << endl;
            pClient->SetConnectionState(ConnectionState::ERROR, evutil_socket_error_to_string(err));
        } else if (events & BEV_EVENT_TIMEOUT)
        {
            cerr << "Connection timeout" << endl;
            pClient->SetConnectionState(ConnectionState::TIMEOUT);
        }
    }
}

void JsonRpcClient::EnterEventLoop()
{
    event_base_dispatch(m_eventBase);
}

void JsonRpcClient::BreakEventLoop()
{
    event_base_loopbreak(m_eventBase);
}

void JsonRpcClient::OnDataReceived(struct bufferevent* bev, void* pContext)
{
    auto* pClient = static_cast<JsonRpcClient*>(pContext);
    struct evbuffer* pInput = bufferevent_get_input(bev);
    const size_t nLength = evbuffer_get_length(pInput);

    // Append the received data to the buffer
    pClient->m_sResponseBuffer.resize(pClient->m_sResponseBuffer.size() + nLength);
    evbuffer_copyout(pInput, &pClient->m_sResponseBuffer[pClient->m_sResponseBuffer.size() - nLength], nLength);
    
    evbuffer_drain(pInput, nLength);

    cout << pClient->m_sResponseBuffer << endl;

    // Process complete JSON-RPC responses from the buffer
    size_t nPos;
    string sResponse;
    string error;
    while ((nPos = pClient->m_sResponseBuffer.find("\n")) != std::string::npos)
    {
        sResponse = pClient->m_sResponseBuffer.substr(0, nPos);
        pClient->m_sResponseBuffer.erase(0, nPos + 1);

        // Parse the response as JSON object
        error.clear();
        pClient->m_responseJson = ParseJsonObject(sResponse, error);
        if (!pClient->m_responseJson.has_value())
        {
            if (pClient->GetConnectionState() == ConnectionState::TIMEOUT)
            {
                if (pClient->m_errorCallback)
                    pClient->m_errorCallback(error);
            } else
                // Incomplete JSON response, keep the data in the buffer for the next iteration
                pClient->m_sResponseBuffer.insert(0, sResponse + "\n");

            pClient->BreakEventLoop();
            break;
        }

        // Invoke the response callback with the received response
        if (pClient->m_responseCallback && !pClient->m_bSyncRequestPending)
            pClient->m_responseCallback(*pClient->m_responseJson);

        pClient->BreakEventLoop();
    }    
}

optional<nlohmann::json> JsonRpcClient::ParseJsonObject(const std::string& sJson, string &error)
{
    try
    {
        return nlohmann::json::parse(sJson);
    }
    catch (const nlohmann::json::parse_error& e)
    {
        // Incomplete or invalid JSON
        error = e.what();
        return nullopt;
    }
}
