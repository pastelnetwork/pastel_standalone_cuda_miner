// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstdint>
// RPC error codes
enum class RPC_ERROR_CODE : int
{
    //! Standard JSON-RPC 2.0 errors
	INVALID_REQUEST       = -32600, // Invalid JSON was received by the server. An error occurred on the server while parsing the JSON text.
	METHOD_NOT_FOUND      = -32601, // The method does not exist / is not available.
	INVALID_PARAMS        = -32602, // Invalid method parameter(s).
	INTERNAL              = -32603, // Internal JSON-RPC error.
    PARSE                 = -32700, // parse error

    //! General application defined errors
    MISC                  = -1,  //! std::exception thrown in command handling
    TYPE                  = -3,  //! Unexpected type was passed as parameter
    OUT_OF_MEMORY         = -7,  //! Ran out of memory during operation
    INVALID_PARAMETER     = -8,  //! Invalid, missing or duplicate parameter
    DESERIALIZATION       = -22, //! Error parsing or validating structure in raw format
	POOL_ERROR            = -50, //! Error returned by the pool
};

inline bool is_rpc_error_code_valid(const int e)
{
    return (e >= -32700 && e <= -32600) || (e == -32700) || (e >= -22 && e <= -1);
}