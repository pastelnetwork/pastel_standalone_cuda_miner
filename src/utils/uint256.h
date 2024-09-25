// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstring>
#include <algorithm>

#include <local_types.h>

template<unsigned int BITS>
class base_blob
{
protected:
    enum { WIDTH = BITS / 8 };
    alignas(uint32_t) uint8_t data[WIDTH];

public:
    inline static constexpr size_t SIZE = WIDTH;
    inline static constexpr size_t STR_SIZE = WIDTH * 2;
	inline static constexpr size_t WORD_SIZE = SIZE / 4;

    base_blob() noexcept
    {
        memset(data, 0, sizeof(data));
    }

    explicit base_blob(const v_uint8& vch)
	{
		memset(data, 0, sizeof(data));
		if (vch.size() <= SIZE)
#ifdef _MSC_VER
			memcpy_s(data, sizeof(data), vch.data(), vch.size());
#else
			memcpy(data, vch.data(), vch.size());
#endif
	}

    base_blob(base_blob && b) noexcept
    {
        std::swap(data, b.data);
    }
    base_blob& operator=(base_blob && b) noexcept
    {
        if (this != &b)
            std::swap(data, b.data);
        return *this;
    }
    base_blob(const base_blob &b) noexcept
    {
#ifdef _MSC_VER
        memcpy_s(data, sizeof(data), b.data, sizeof(b.data));
#else
        memcpy(data, b.data, sizeof(data));
#endif
    }
    base_blob& operator=(const base_blob &b) noexcept
    {
        if (this != &b)
        {
#ifdef _MSC_VER
            memcpy_s(data, sizeof(data), b.data, sizeof(b.data));
#else
            memcpy(data, b.data, sizeof(data));
#endif
        }
        return *this;
    }

    bool IsNull() const noexcept
    {
        return std::all_of(data, data + WIDTH, [](const uint8_t x) { return x == 0; });
    }

    void SetNull() noexcept
    {
        memset(data, 0, sizeof(data));
    }

    friend inline bool operator==(const base_blob& a, const base_blob& b) { return memcmp(a.data, b.data, sizeof(a.data)) == 0; }
    friend inline bool operator!=(const base_blob& a, const base_blob& b) { return memcmp(a.data, b.data, sizeof(a.data)) != 0; }
    friend inline bool operator<(const base_blob& a, const base_blob& b) { return memcmp(a.data, b.data, sizeof(a.data)) < 0; }

    std::string GetHex() const;
    void SetHex(const char* psz);
    void SetHex(const std::string& str);
    std::string ToString() const;

    unsigned char* begin() noexcept
    {
        return &data[0];
    }

    unsigned char* end() noexcept
    {
        return &data[WIDTH];
    }

    const unsigned char* begin() const noexcept
    {
        return &data[0];
    }

    const unsigned char* end() const noexcept
    {
        return &data[WIDTH];
    }

    const unsigned char* cbegin() const noexcept
    {
        return &data[0];
    }

    const unsigned char* cend() const noexcept
    {
        return &data[WIDTH];
    }

    unsigned int size() const noexcept
    {
        return sizeof(data);
    }

    template<typename Stream>
    void Serialize(Stream& s) const
    {
        s.write((char*)data, sizeof(data));
    }

    template<typename Stream>
    void Unserialize(Stream& s)
    {
        s.read((char*)data, sizeof(data));
    }

    // set Nth word in the blob
    void SetUint32(size_t n, const uint32_t x) noexcept;

    // Reverse the bytes in the blob
    base_blob& Reverse() noexcept
    {
        std::reverse(data, data + WIDTH);
        return *this;
    }
};

/** 256-bit opaque blob.
 * @note This type is called uint256 for historical reasons only. It is an
 * opaque blob of 256 bits and has no integer operations. Use arith_uint256 if
 * those are required.
 */
class uint256 : public base_blob<256>
{
public:
    uint256() noexcept {}
    uint256(const base_blob<256>& b) noexcept : 
        base_blob<256>(b)
    {}
    explicit uint256(const v_uint8& vch) noexcept : 
        base_blob<256>(vch)
    {}

    /** A cheap hash function that just returns 64 bits from the result, it can be
     * used when the contents are considered uniformly random. It is not appropriate
     * when the value can easily be influenced from outside as e.g. a network adversary could
     * provide values to trigger worst-case behavior.
     * @note The result of this function is not stable between little and big endian.
     */
    uint64_t GetCheapHash() const noexcept
    {
        uint64_t result;
        memcpy((void*)&result, (void*)data, sizeof(uint64_t));
        return result;
    }

    /** A more secure, salted hash function.
     * @note This hash is not stable between little and big endian.
     */
    uint64_t GetHash(const uint256& salt) const noexcept;
};

static void inline HashMix(uint32_t& a, uint32_t& b, uint32_t& c)
{
    // Taken from lookup3, by Bob Jenkins.
    a -= c;
    a ^= ((c << 4) | (c >> 28));
    c += b;
    b -= a;
    b ^= ((a << 6) | (a >> 26));
    a += c;
    c -= b;
    c ^= ((b << 8) | (b >> 24));
    b += a;
    a -= c;
    a ^= ((c << 16) | (c >> 16));
    c += b;
    b -= a;
    b ^= ((a << 19) | (a >> 13));
    a += c;
    c -= b;
    c ^= ((b << 4) | (b >> 28));
    b += a;
}

static void inline HashFinal(uint32_t& a, uint32_t& b, uint32_t& c)
{
    // Taken from lookup3, by Bob Jenkins.
    c ^= b;
    c -= ((b << 14) | (b >> 18));
    a ^= c;
    a -= ((c << 11) | (c >> 21));
    b ^= a;
    b -= ((a << 25) | (a >> 7));
    c ^= b;
    c -= ((b << 16) | (b >> 16));
    a ^= c;
    a -= ((c << 4) | (c >> 28));
    b ^= a;
    b -= ((a << 14) | (a >> 18));
    c ^= b;
    c -= ((b << 24) | (b >> 8));
}


/* uint256 from const char *.
 * This is a separate function because the constructor uint256(const char*) can result
 * in dangerously catching uint256(0).
 */
inline uint256 uint256S(const char *str)
{
    uint256 rv;
    rv.SetHex(str);
    return rv;
}

/* uint256 from std::string.
 * This is a separate function because the constructor uint256(const std::string &str) can result
 * in dangerously catching uint256(0) via std::string(const char*).
 */
inline uint256 uint256S(const std::string& str)
{
    uint256 rv;
    rv.SetHex(str);
    return rv;
}

// convert hex-encoded string to uint256 with error checking
bool parse_uint256(std::string& error, uint256& value, const std::string &sUint256, const char *szValueDesc = nullptr);

extern template class base_blob<256>;
