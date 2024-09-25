// Copyright (c) 2024 The Pastfel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <iostream>
#include <memory>
#include <chrono>

#include <compat.h>
#include <event2/buffer.h>
#include <event2/bufferevent.h>
#include <event2/event.h>
#include <scope_guard.hpp>
#include <tinyformat.h>

#include <src/utils/enum_util.h>
#include <src/utils/logger.h>
#include <src/stratum/json-rpc-client.h>

using namespace std;
using namespace std::chrono_literals;
using json = nlohmann::json;

#ifdef _DEBUG
constexpr auto JSON_RPC_RESPONSE_TIMEOUT_SECS = 300s;
#else
constexpr auto JSON_RPC_RESPONSE_TIMEOUT_SECS = 15s;
#endif

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
	nCode(0),
    message(message),
    data(nullptr),
    err(to_string(to_integral_type(code)) + ": " + message)
{}

JsonRpcException::JsonRpcException(RPC_ERROR_CODE code, const std::string &message, const json &data) noexcept :
    code(code),
	nCode(0),
    message(message),
    data(data),
    err(to_string(to_integral_type(code)) + ": " + message + ", data: " + data.dump())
{}

JsonRpcException::JsonRpcException(const string& message, const int nErrorCode) noexcept :
	code(RPC_ERROR_CODE::POOL_ERROR),
	nCode(nErrorCode),
	message(message),
	data(nullptr)
{
	err = to_string(nErrorCode) + ": " + message;
}

JsonRpcClient::JsonRpcClient(const char *szThreadName) :
	CStoppableServiceThread(szThreadName),
    m_rpcVersion(rpcVersion::v2),
    m_nServerPort(0),
    m_pEventBase(event_base_new()),
    m_bufferEvent(nullptr),
    m_connectionState(ConnectionState::SRV_DISCONNECTED),
	m_bBreakEventLoop(false)
{
    m_sResponseBuffer.reserve(2048);
}

JsonRpcClient::~JsonRpcClient()
{
    FreeConnection();
    if (m_pEventBase)
    {
        event_base_free(m_pEventBase);
		m_pEventBase = nullptr;
    }
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

void JsonRpcClient::setServerInfo(const std::string& sServerAddress, unsigned short nServerPort)
{
    m_sServerAddress = sServerAddress;
    m_nServerPort = nServerPort;
}

bool JsonRpcClient::Connect()
{
    FreeConnection();

    m_bufferEvent = bufferevent_socket_new(m_pEventBase, -1, BEV_OPT_CLOSE_ON_FREE | BEV_OPT_THREADSAFE);
    if (!m_bufferEvent)
    {
		gl_console_logger->error("Failed to create buffer event");
        return false;
    }

    // Set read/write timeouts
    // struct timeval timeout;
    // timeout.tv_sec = JSON_RPC_RESPONSE_TIMEOUT_SECS.count(); // Set timeout to 10 seconds
    // timeout.tv_usec = 0;
    // bufferevent_set_timeouts(m_bufferEvent, &timeout, &timeout);

    if (bufferevent_socket_connect_hostname(m_bufferEvent, nullptr, AF_INET, m_sServerAddress.c_str(), m_nServerPort) < 0)
    {
		gl_console_logger->error("Failed to connect to [{}:{}]", m_sServerAddress, m_nServerPort);
        bufferevent_free(m_bufferEvent);
        return false;
    }

    bufferevent_setcb(m_bufferEvent, bev_callback, nullptr, OnEventCallback, this);
    bufferevent_enable(m_bufferEvent, EV_READ | EV_WRITE);
    return true;
}

void JsonRpcClient::SetConnectionState(const ConnectionState state, const char *szError) noexcept
{
    m_connectionState = state;
    if (szError)
        m_sError = szError;
    if (m_connectionState == ConnectionState::SRV_DISCONNECTED)
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
        const bool bSucceeded = SendAndReceive(error, j.dump(), id);
        if (!bSucceeded)
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, error);

        if (!m_responseJson.has_value())
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, "Json-RPC response is empty for " + name);

        const json &response = *m_responseJson;
        if (json_rpc_has_key_type(response, "error", json::value_t::object))
            throw JsonRpcException::fromJson(response["error"]);

        if (json_rpc_has_key_type(response, "error", json::value_t::string))
            throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, response["error"]);

        if (json_rpc_has_key_type(response, "error", json::value_t::array))
        {
			auto& error_response = response["error"];
            if (error_response.size() == 2)
                throw JsonRpcException(error_response[1].get<string>(), error_response[0].get<int>());
        }

        if (json_rpc_has_key(response, "result") && json_rpc_has_key(response, "id"))
        {
			const auto& jResult = response["result"];
            if (response["id"].type() == json::value_t::string)
				return JsonRpcResponse { response["id"].get<string>(), jResult };

            return JsonRpcResponse{ response["id"].get<int>(), jResult };
        }

        throw JsonRpcException(RPC_ERROR_CODE::INTERNAL, R"(invalid server response: neither "result" nor "error" fields found)");
    } catch (json::parse_error &e) {
        throw JsonRpcException(RPC_ERROR_CODE::PARSE, std::string("invalid JSON response from server: ") + e.what());
    }
}

bool JsonRpcClient::addSyncRequestId(const id_type requestId)
{
	lock_guard<mutex> lock(m_IdMutex);
    // check for monostate
	if (get_if<monostate>(&requestId) != nullptr)
		return false;
	m_IdSent.insert(requestId);
	return true;
}

bool JsonRpcClient::removeSyncRequestId(const id_type requestId)
{
	lock_guard<mutex> lock(m_IdMutex);
    // check for monostate
	if (get_if<monostate>(&requestId) != nullptr)
		return false;
    m_IdSent.erase(requestId);
    return true;
}

string to_string(const id_type& id)
{
    return visit([](auto&& arg) -> string {
        using T = decay_t<decltype(arg)>;
        if constexpr (is_same_v<T, monostate>) {
            return ""; // Return an empty string for monostate
        } else if constexpr (is_same_v<T, int>) {
            return to_string(arg); // Convert int to string
        } else if constexpr (is_same_v<T, string>) {
            return arg; // Return the string directly
        }
    }, id);
}

bool JsonRpcClient::handleReceivedId(const id_type &requestId)
{
    if (get_if<monostate>(&requestId) != nullptr)
        return false;

    {
        lock_guard<mutex> lock(m_IdMutex);
        auto it = m_IdSent.find(requestId);
        if (it == m_IdSent.cend())
        {
			gl_console_logger->error("Received response for unknown request id [{}]", to_string(requestId));
            return false;
        }
        m_IdSent.erase(requestId);
		m_IdReceived.insert(requestId);
    }

	// Notify the waiting thread that the response has been received
	lock_guard<mutex> lock(m_mutex);
	m_cvSyncRequest.notify_all();
	return true;
}

JsonRpcNotify JsonRpcClient::ParseJsonRpcNotify(const json &j)
{
    JsonRpcNotify notify;

    // Extract 'id' field
    if (!j.contains("id"))
        throw JsonRpcException(RPC_ERROR_CODE::INVALID_REQUEST,
            "JsonRpcRequest must contain an 'id' field.");

    if (j["id"].is_null())
        notify.id = monostate{};
    else if (j["id"].is_number_integer())
        notify.id = j["id"].get<int>();
    else if (j["id"].is_string())
        notify.id = j["id"].get<std::string>();
    else
        throw JsonRpcException(RPC_ERROR_CODE::INVALID_REQUEST, 
            "JsonRpcRequest 'id' field must be either null, integer or string.");

	// check for result field
	notify.isResult = j.contains("result");

    // check for error field
    if (j.contains("error"))
    {
		const auto& jError = j["error"];
        if (!jError.is_null())
        {
            if (jError.is_string())
                notify.error = jError.get<std::string>();
        }
    }

    if (!notify.isResult)
    {
        // Extract 'method' field
        const auto& jMethod = j["method"];
        if (!jMethod.is_string())
            throw JsonRpcException(RPC_ERROR_CODE::METHOD_NOT_FOUND,
                "JsonRpcRequest must contain a 'method' field of string type.");
        notify.method = jMethod.get<std::string>();

        // Extract 'params' field
        if (j.contains("params"))
            notify.params = j["params"];
        else
            notify.params = json::object();
    }
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
	gl_console_logger->info(request);
    if (bufferevent_write(m_bufferEvent, request.c_str(), request.size()) < 0 ||
        bufferevent_write(m_bufferEvent, "\n", 1) < 0)
    {
        error = "Failed to send request";
        if (!m_sError.empty())
            error += ": " + m_sError;
        return false;
    }

    return true;
}

bool JsonRpcClient::SendAndReceive(string& error, const string& request, const id_type requestId)
{
	if (!Send(error, request))
		return false;

	if (!addSyncRequestId(requestId))
	{
		error = "Invalid request id";
		return false;
	}

    // Wait for the full json-rpc response, it will be processed in bev_callback
	// callback will be called when the response is received
    bool bResponseForIdReceived = false;
	while (!bResponseForIdReceived)
    {
		unique_lock<mutex> lock(m_mutex);
		if (m_cvSyncRequest.wait_for(lock, JSON_RPC_RESPONSE_TIMEOUT_SECS) == cv_status::timeout)
		{
			error = "Timeout waiting for response";
			return false;
		}
        const auto state = GetConnectionState();
        if (state == ConnectionState::SRV_DISCONNECTED ||
            state == ConnectionState::SRV_ERROR)
        {
            // connection is broken - no point to wait for the response
            removeSyncRequestId(requestId);
            break;
        }
		// check if the response for the request id has been received
		// remove from the set of received request ids
        {
			lock_guard<mutex> lock(m_IdMutex);
			auto it = m_IdReceived.find(requestId);
            if (it != m_IdReceived.end())
            {
				bResponseForIdReceived = true;
				m_IdReceived.erase(it);
				gl_console_logger->debug("Response received for request id [{}]", to_string(requestId));
                break;
            }
        }
    }
	return true;
}

void JsonRpcClient::OnEventCallback(struct bufferevent* bev, short events, void* pContext)
{
    auto* pClient = static_cast<JsonRpcClient*>(pContext);
    if (events & BEV_EVENT_CONNECTED)
    {
        pClient->SetConnectionState(ConnectionState::SRV_CONNECTED);
    }
    else if (events & (BEV_EVENT_ERROR | BEV_EVENT_EOF | BEV_EVENT_TIMEOUT))
    {
        if (events & BEV_EVENT_EOF)
        {
			gl_console_logger->error("Connection closed by the server");
            pClient->SetConnectionState(ConnectionState::SRV_DISCONNECTED);
        } else if (events & BEV_EVENT_ERROR)
        {
            int err = EVUTIL_SOCKET_ERROR();
			gl_console_logger->error("Socket error: {}", evutil_socket_error_to_string(err));
            pClient->SetConnectionState(ConnectionState::SRV_ERROR, evutil_socket_error_to_string(err));
        } else if (events & BEV_EVENT_TIMEOUT)
        {
			gl_console_logger->error("Connection timeout");
            pClient->SetConnectionState(ConnectionState::SRV_TIMEOUT);
        }
    }
}

void JsonRpcClient::execute()
{
    try
    {
        while (!m_bBreakEventLoop)
        {
            const auto state = GetConnectionState();
			if (state == ConnectionState::SRV_DISCONNECTED ||
                state == ConnectionState::SRV_ERROR)
			{
				if (!Connect())
				{
					gl_console_logger->error("Failed to connect to the server");
					this_thread::sleep_for(10s);
					continue;
				}
			}

            gl_console_logger->info("Entered event base loop");
			m_bBreakEventLoop = false;
            int result = event_base_loop(m_pEventBase, EVLOOP_NO_EXIT_ON_EMPTY);
            if (result < 0)
            {
				gl_console_logger->error("Error in event loop: {}", strerror(errno));
                break; // Break the loop if there is an error
            }
			gl_console_logger->info("Exited event base loop");
            this_thread::sleep_for(5s);
        }

	} catch (const std::exception& e)
	{
		gl_console_logger->error("Exception in JsonRpcClient: {}", e.what());
	} catch (...)
	{
		gl_console_logger->error("Unknown exception in JsonRpcClient");
	}
}

void JsonRpcClient::BreakEventLoop()
{
	m_bBreakEventLoop = true;
    event_base_loopbreak(m_pEventBase);
}

void JsonRpcClient::bev_callback(struct bufferevent* bev, void* pContext)
{
    auto* pClient = static_cast<JsonRpcClient*>(pContext);
    struct evbuffer* pInput = bufferevent_get_input(bev);
    const size_t nLength = evbuffer_get_length(pInput);

    // Append the received data to the buffer
    pClient->m_sResponseBuffer.resize(pClient->m_sResponseBuffer.size() + nLength);
    evbuffer_copyout(pInput, &pClient->m_sResponseBuffer[pClient->m_sResponseBuffer.size() - nLength], nLength);
    
    evbuffer_drain(pInput, nLength);

    gl_console_logger->info(pClient->m_sResponseBuffer);

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
            if (pClient->GetConnectionState() == ConnectionState::SRV_TIMEOUT)
            {
                if (pClient->m_errorCallback)
                    pClient->m_errorCallback(error);
            } else
                // Incomplete JSON response, keep the data in the buffer for the next iteration
                pClient->m_sResponseBuffer.insert(0, sResponse + "\n");
            break;
        }

        // Invoke the response callback with the received response
        if (pClient->m_responseCallback)
            pClient->m_responseCallback(*pClient->m_responseJson);
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
