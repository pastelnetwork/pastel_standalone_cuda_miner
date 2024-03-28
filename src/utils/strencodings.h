// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <string>

#include <local_types.h>

signed char HexDigit(const char c) noexcept;

template<typename T>
std::string HexStr(const T itbegin, const T itend, bool fSpaces=false)
{
    std::string rv;
    static const char hexmap[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                                     '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
    rv.reserve((itend - itbegin) * (fSpaces ? 3 : 2));
    for(T it = itbegin; it < itend; ++it)
    {
        unsigned char val = (unsigned char)(*it);
        if (fSpaces && it != itbegin)
            rv.push_back(' ');
        rv.push_back(hexmap[val >> 4]);
        rv.push_back(hexmap[val & 15]);
    }

    return rv;
}

template<typename T>
inline std::string HexStr(const T& vch, bool fSpaces=false)
{
    return HexStr(vch.cbegin(), vch.cend(), fSpaces);
}

bool IsHex(const std::string& str) noexcept;

v_uint8 ParseHex(const char* psz);
v_uint8 ParseHex(const std::string& str);

