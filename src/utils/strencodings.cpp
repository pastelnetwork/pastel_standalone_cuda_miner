// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstddef>
#include <cstring>
#include <stdexcept>

#include <compat/endian.h>
#include <src/utils/str_utils.h>
#include <src/utils/strencodings.h>

using namespace std;

static constexpr signed char p_util_hexdigit[] =
{ -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
   0,1,2,3,4,5,6,7,8,9,-1,-1,-1,-1,-1,-1,
  -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,0xa,0xb,0xc,0xd,0xe,0xf,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, };

signed char HexDigit(const char c) noexcept
{
    return p_util_hexdigit[static_cast<const unsigned char>(c)];
}

bool IsHex(const string& str) noexcept
{
    for (const char &ch : str)
    {
        if (HexDigit(ch) < 0)
            return false;
    }
    return (str.size() > 0) && (str.size() % 2 == 0);
}

v_uint8 ParseHex(const char* psz)
{
    // convert hex dump to vector
    v_uint8 vch;
    if (!psz)
        return vch;
    const size_t nLength = psz ? strlen(psz) : 0;
    vch.reserve(nLength / 2);
    while (*psz)
    {
        while (isspaceex(*psz))
            psz++;
        signed char c = HexDigit(*psz);
        psz++;
        if (c == static_cast<signed char>(-1))
            break;
        unsigned char n = (c << 4);
        c = HexDigit(*psz);
        psz++;
        if (c == (signed char)-1)
            break;
        n |= c;
        vch.push_back(n);
    }
    return vch;
}

v_uint8 ParseHex(const string& str)
{
    return ParseHex(str.c_str());
}

string HexStr(const uint32_t n, bool bSpaces)
{
    const auto* bytes = reinterpret_cast<const unsigned char*>(&n);
    return HexStr(bytes, bytes + sizeof(n), bSpaces);
}

v_uint8 DecodeBase64(const char* p, bool* pfInvalid)
{
    static constexpr int decode64_table[] =
    {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1,
        -1, -1, -1, -1, -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
        15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26, 27, 28,
        29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
    };

    const char* e = p;
    v_uint8 val;
    val.reserve(strlen(p));
    while (*p)
    {
        const int x = decode64_table[static_cast<unsigned char>(*p)];
        if (x == -1)
            break;
        val.push_back(x);
        ++p;
    }

    v_uint8 ret;
    ret.reserve((val.size() * 3) / 4);
    bool bValid = ConvertBits<6, 8, false>([&](unsigned char c)
        {
            ret.push_back(c);
        }, 
        val.begin(), val.end());

    const char* q = p;
    while (bValid && *p)
    {
        if (*p != '=')
        {
            bValid = false;
            break;
        }
        ++p;
    }
    bValid &= (p - e) % 4 == 0 && p - q < 4;
    if (pfInvalid) 
        *pfInvalid = !bValid;

    return ret;
}

/**
 * Decode base64 encoded string.
 *  
 * \param str - base64 encoded string
 * \param pfInvalid - pointer to bool, set to true if there was an error decoding string
 * \return decoded string, may be partial if there was a failure
 */
string DecodeBase64(const string& str, bool* pfInvalid)
{
    return vector_to_string(DecodeBase64(str.c_str(), pfInvalid));
}

string EncodeBase64(const unsigned char* pch, size_t len)
{
    static constexpr auto PBASE64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string str;
    str.reserve(((len + 2) / 3) * 4);
    ConvertBits<8, 6, true>([&](int v)
    {
        str += PBASE64[v];
    }, pch, pch + len);
    while (str.size() % 4)
        str += '=';
    return str;
}

string EncodeBase64(const string& str)
{
    return EncodeBase64((const unsigned char*)str.c_str(), str.size());
}

uint8_t ConvertHexToUint8LE(const string& sHexStr)
{
    if (sHexStr.size() != 2)
        throw invalid_argument("Hex string must be 2 characters long for uint8_t.");

    return static_cast<uint8_t>(stoul(sHexStr, nullptr, 16));  // No byte swap needed for 1-byte value
}

uint16_t ConvertHexToUint16LE(const string& sHexStr)
{
    if (sHexStr.size() != 4)
        throw invalid_argument("Hex string must be 4 characters long for uint16_t.");

    uint16_t nValue = static_cast<uint16_t>(stoul(sHexStr, nullptr, 16));
    return bswap_16(nValue);
}

uint32_t ConvertHexToUint32LE(const string& sHexStr)
{
    if (sHexStr.size() != 8)
        throw invalid_argument("Hex string must be 8 characters long for uint32_t.");

    uint32_t nValue = static_cast<uint32_t>(stoul(sHexStr, nullptr, 16));
    return bswap_32(nValue);
}

uint64_t ConvertHexToUint64LE(const string& sHexStr)
{
    if (sHexStr.size() != 16)
        throw invalid_argument("Hex string must be 16 characters long for uint64_t.");

    uint64_t value = stoull(sHexStr, nullptr, 16);
    return bswap_64(value);
}
