// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <string>

#include <tinyformat.h>

#include <compat/endian.h>
#include <src/utils/uint256.h>
#include <src/utils/strencodings.h>

using namespace std;

template <unsigned int BITS>
string base_blob<BITS>::GetHex() const
{
    char psz[sizeof(data) * 2 + 1];
    for (unsigned int i = 0; i < sizeof(data); i++)
        sprintf(psz + i * 2, "%02x", data[sizeof(data) - i - 1]);
    return string(psz, psz + sizeof(data) * 2);
}

template <unsigned int BITS>
void base_blob<BITS>::SetHex(const char* psz)
{
    memset(data, 0, sizeof(data));

    // skip leading spaces
    while (isspace(*psz))
        psz++;

    // skip 0x
    if (psz[0] == '0' && tolower(psz[1]) == 'x')
        psz += 2;

    // hex string to uint
    const char* pbegin = psz;
    while (::HexDigit(*psz) != -1)
        psz++;
    psz--;
    unsigned char* p1 = (unsigned char*)data;
    unsigned char* pend = p1 + WIDTH;
    while (psz >= pbegin && p1 < pend)
    {
        *p1 = ::HexDigit(*psz--);
        if (psz >= pbegin)
        {
            *p1 |= ((unsigned char)::HexDigit(*psz--) << 4);
            p1++;
        }
    }
}

template <unsigned int BITS>
void base_blob<BITS>::SetHex(const string& str)
{
    SetHex(str.c_str());
}

template <unsigned int BITS>
string base_blob<BITS>::ToString() const
{
    return GetHex();
}

    // set Nth word in the blob
template<unsigned int BITS>
void base_blob<BITS>::SetUint32(size_t n, const uint32_t x) noexcept
{
    assert(n < WORD_SIZE);
    uint32_t* p = reinterpret_cast<uint32_t*>(data);
    p[n] = htole32(x);
}

uint64_t uint256::GetHash(const uint256& salt) const noexcept
{
    uint32_t a, b, c;
    const uint32_t *pn = (const uint32_t*)data;
    const uint32_t *salt_pn = (const uint32_t*)salt.data;
    a = b = c = 0xdeadbeef + WIDTH;

    a += pn[0] ^ salt_pn[0];
    b += pn[1] ^ salt_pn[1];
    c += pn[2] ^ salt_pn[2];
    HashMix(a, b, c);
    a += pn[3] ^ salt_pn[3];
    b += pn[4] ^ salt_pn[4];
    c += pn[5] ^ salt_pn[5];
    HashMix(a, b, c);
    a += pn[6] ^ salt_pn[6];
    b += pn[7] ^ salt_pn[7];
    HashFinal(a, b, c);

    return ((((uint64_t)b) << 32) | c);
}

/**
 * Convert string to uint256 with error checking.
 * 
 * \param error - return error if any
 * \param hash - converted uint256
 * \param sUint256 - input uint256 value string
 * \param szValueDesc - optional value description (to form an error message)
 *  
 * \return true if string was successfully converted to uint256
 */
bool parse_uint256(string& error, uint256& value, const string &sUint256, const char *szValueDesc)
{
    bool bRet = false;
    do
    {
        // validate string size
        if (sUint256.size() != uint256::STR_SIZE)
        {
            error = strprintf("Incorrect %s value size: %zu, expected: %zu. [%s]",
                szValueDesc ? szValueDesc : "uint256", sUint256.size(), uint256::SIZE * 2, sUint256);
            break;
        }
        if (!IsHex(sUint256))
        {
            error = strprintf("Invalid %s hexadecimal value: %s",
                szValueDesc ? szValueDesc : "uint256", sUint256);
            break;
        }

        value = uint256S(sUint256);
        bRet = true;
    } while (false);
    return bRet;
}

template class base_blob<256>;