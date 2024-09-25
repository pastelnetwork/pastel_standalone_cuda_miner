// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <cstring>

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

/**
 * Convert from one power-of-2 number base to another.
 *
 * Examples using ConvertBits<8, 5, true>():
 * 000000 -> 0000000000
 * 202020 -> 0400100200
 * 757575 -> 0e151a170a
 * abcdef -> 150f061e1e
 * ffffff -> 1f1f1f1f1e
 */
template<int frombits, int tobits, bool pad, typename O, typename I>
bool ConvertBits(const O& outfn, I it, I end)
{
    size_t acc = 0;
    size_t bits = 0;
    constexpr size_t maxv = (1 << tobits) - 1;
    constexpr size_t max_acc = (1 << (frombits + tobits - 1)) - 1;
    while (it != end)
    {
        acc = ((acc << frombits) | *it) & max_acc;
        bits += frombits;
        while (bits >= tobits)
        {
            bits -= tobits;
            outfn((acc >> bits) & maxv);
        }
        ++it;
    }
    if (pad)
    {
        if (bits)
            outfn((acc << (tobits - bits)) & maxv);
    }
    else if (bits >= frombits || ((acc << (tobits - bits)) & maxv))
        return false;
    return true;
}

/**
 * Converts byte vector to string.
 * 
 * \param v - byte vector
 * \return output string
 */
inline std::string vector_to_string(const v_uint8 &v)
{
    std::string s;
    s.resize(v.size());
    memcpy(s.data(), v.data(), v.size());
    return s;
}

template<typename T>
inline std::string HexStr(const T& vch, bool fSpaces = false)
{
    return HexStr(vch.cbegin(), vch.cend(), fSpaces);
}

std::string HexStr(const uint32_t n, bool bSpaces = false);

bool IsHex(const std::string& str) noexcept;

v_uint8 ParseHex(const char* psz);
v_uint8 ParseHex(const std::string& str);

v_uint8 DecodeBase64(const char* p, bool* pfInvalid = nullptr);
std::string DecodeBase64(const std::string& str, bool* pfInvalid = nullptr);

std::string EncodeBase64(const unsigned char* pch, size_t len);
std::string EncodeBase64(const std::string& str);

uint8_t ConvertHexToUint8LE(const std::string& sHexStr);
uint16_t ConvertHexToUint16LE(const std::string& sHexStr);
uint32_t ConvertHexToUint32LE(const std::string& sHexStr);
uint64_t ConvertHexToUint64LE(const std::string& sHexStr);
