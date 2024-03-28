// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstdint>
// RPC error codes
enum class RPC_ERROR_CODE : int
{
    //! Standard JSON-RPC 2.0 errors
    INVALID_REQUEST       = -32600,
    METHOD_NOT_FOUND      = -32601,
    INVALID_PARAMS        = -32602,
    INTERNAL              = -32603,
    PARSE                 = -32700,

    //! General application defined errors
    MISC                  = -1,  //! std::exception thrown in command handling
    TYPE                  = -3,  //! Unexpected type was passed as parameter
    OUT_OF_MEMORY         = -7,  //! Ran out of memory during operation
    INVALID_PARAMETER     = -8,  //! Invalid, missing or duplicate parameter
    DESERIALIZATION       = -22, //! Error parsing or validating structure in raw format
};

inline bool is_rpc_error_code_valid(const int e)
{
    return (e >= -32700 && e <= -32600) || (e == -32700) || (e >= -22 && e <= -1);
}