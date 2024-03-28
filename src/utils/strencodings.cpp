// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <cstdint>
#include <cstddef>
#include <cstring>

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

inline signed char HexDigit(const char c) noexcept
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
    const size_t nLength = psz ? strlen(psz) : 0;
    vch.reserve(nLength / 2);
    while (psz)
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
