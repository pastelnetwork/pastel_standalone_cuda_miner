// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cinttypes>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <vector>
#include <string>
#include <algorithm>
#include <memory>
#include <ios>

#include <compat/endian.h>

#include <tinyformat.h>

class CSizeComputer;

enum
{
    // primary actions
    SER_NETWORK         = (1 << 0), // for communication between nodes
    SER_DISK            = (1 << 1), // for disk storage
    SER_GETHASH         = (1 << 2), // to get hash of object
};

#define READWRITE(obj)                  (::SerReadWrite(s, (obj), ser_action))
#define READWRITE_CHECKED(obj, maxsize) (::SerReadWriteChecked(s, (obj), ser_action, maxsize))

/**
 * Used to bypass the rule against non-const reference to temporary
 * where it makes sense with wrappers such as CFlatData or CTxDB
 */
template<typename T>
inline T& REF(const T& val)
{
    return const_cast<T&>(val);
}

/**
 * Used to acquire a non-const pointer "this" to generate bodies
 * of const serialization operations from a template
 */
template<typename T>
inline T* NCONST_PTR(const T* val)
{
    return const_cast<T*>(val);
}

/** 
 * Implement three methods for serializable objects. These are actually wrappers over
 * "SerializationOp" template, which implements the body of each class' serialization
 * code. Adding "ADD_SERIALIZE_METHODS" in the body of the class causes these wrappers to be
 * added as members. 
 */
#define ADD_SERIALIZE_METHODS                                         \
    template<typename Stream>                                         \
    void Serialize(Stream& s) const { NCONST_PTR(this)->SerializationOp(s, SERIALIZE_ACTION::Write); } \
    template<typename Stream>                                         \
    void Unserialize(Stream& s) { SerializationOp(s, SERIALIZE_ACTION::Read); }


/**
 * network protocol versioning
 */

inline constexpr int PROTOCOL_VERSION = 170011;

/**
 * Compact Size
 * size <  253        -- 1 byte
 * size <= 0xFFFF     -- 3 bytes  (253 + 2 bytes)
 * size <= 0xFFFFFFFF -- 5 bytes  (254 + 4 bytes)
 * size >  0xFFFFFFFF -- 9 bytes  (255 + 8 bytes)
 */
inline unsigned int GetSizeOfCompactSize(const uint64_t nSize)
{
    if (nSize < 253)                return 1;
    else if (nSize <= 0xFFFFu)      return 3;
    else if (nSize <= 0xFFFFFFFFu)  return 5;
    else                            return 9;
}

inline void WriteCompactSize(CSizeComputer& os, const uint64_t nSize);

template<typename Stream>
void WriteCompactSize(Stream& os, const uint64_t nSize)
{
    if (nSize < 253)
    {
        ser_writedata8(os, static_cast<uint8_t>(nSize));
    }
    else if (nSize <= 0xFFFFu)
    {
        ser_writedata8(os, 253);
        ser_writedata16(os, static_cast<uint16_t>(nSize));
    }
    else if (nSize <= 0xFFFFFFFFu)
    {
        ser_writedata8(os, 254);
        ser_writedata32(os, static_cast<uint32_t>(nSize));
    }
    else
    {
        ser_writedata8(os, 255);
        ser_writedata64(os, nSize);
    }
}

template<typename Stream>
uint64_t ReadCompactSize(Stream& is, uint64_t max_size = std::numeric_limits<uint64_t>::max())
{
    uint8_t chSize = ser_readdata8(is);
    uint64_t nSizeRet = 0;
    if (chSize < 253)
    {
        nSizeRet = chSize;
    }
    else if (chSize == 253)
    {
        nSizeRet = ser_readdata16(is);
        if (nSizeRet < 253)
            throw std::ios_base::failure("non-canonical ReadCompactSize()");
    }
    else if (chSize == 254)
    {
        nSizeRet = ser_readdata32(is);
        if (nSizeRet < 0x10000u)
            throw std::ios_base::failure("non-canonical ReadCompactSize()");
    }
    else
    {
        nSizeRet = ser_readdata64(is);
        if (nSizeRet < 0x100000000ULL)
            throw std::ios_base::failure("non-canonical ReadCompactSize()");
    }
    if ((max_size < std::numeric_limits<uint64_t>::max()) && (nSizeRet > max_size))
        throw std::ios_base::failure("ReadCompactSize(): size too large");
    return nSizeRet;
}

/*
 * Lowest-level serialization and conversion.
 * @note Sizes of these types are verified in the tests
 */
template<typename Stream> inline void ser_writedata8(Stream &s, uint8_t obj)
{
    s.write((char*)&obj, 1);
}
template<typename Stream> inline void ser_writedata16(Stream &s, uint16_t obj)
{
    obj = htole16(obj);
    s.write((char*)&obj, 2);
}
template<typename Stream> inline void ser_writedata32(Stream &s, uint32_t obj)
{
    obj = htole32(obj);
    s.write((char*)&obj, 4);
}
template<typename Stream> inline void ser_writedata64(Stream &s, uint64_t obj)
{
    obj = htole64(obj);
    s.write((char*)&obj, 8);
}
template<typename Stream> inline uint8_t ser_readdata8(Stream &s)
{
    uint8_t obj;
    s.read((char*)&obj, 1);
    return obj;
}
template<typename Stream> inline uint16_t ser_readdata16(Stream &s)
{
    uint16_t obj;
    s.read((char*)&obj, 2);
    return le16toh(obj);
}
template<typename Stream> inline uint32_t ser_readdata32(Stream &s)
{
    uint32_t obj;
    s.read((char*)&obj, 4);
    return le32toh(obj);
}
template<typename Stream> inline uint64_t ser_readdata64(Stream &s)
{
    uint64_t obj;
    s.read((char*)&obj, 8);
    return le64toh(obj);
}
inline uint64_t ser_double_to_uint64(double x)
{
    union { double x; uint64_t y; } tmp;
    tmp.x = x;
    return tmp.y;
}
inline uint32_t ser_float_to_uint32(float x)
{
    union { float x; uint32_t y; } tmp;
    tmp.x = x;
    return tmp.y;
}
inline double ser_uint64_to_double(uint64_t y)
{
    union { double x; uint64_t y; } tmp;
    tmp.y = y;
    return tmp.x;
}
inline float ser_uint32_to_float(uint32_t y)
{
    union { float x; uint32_t y; } tmp;
    tmp.y = y;
    return tmp.x;
}

template<typename Stream> inline void Serialize(Stream& s, char a    ) { ser_writedata8(s, a); } // TODO Get rid of bare char
template<typename Stream> inline void Serialize(Stream& s, int8_t a  ) { ser_writedata8(s, a); }
template<typename Stream> inline void Serialize(Stream& s, uint8_t a ) { ser_writedata8(s, a); }
template<typename Stream> inline void Serialize(Stream& s, int16_t a ) { ser_writedata16(s, a); }
template<typename Stream> inline void Serialize(Stream& s, uint16_t a) { ser_writedata16(s, a); }
template<typename Stream> inline void Serialize(Stream& s, int32_t a ) { ser_writedata32(s, a); }
template<typename Stream> inline void Serialize(Stream& s, uint32_t a) { ser_writedata32(s, a); }
template<typename Stream> inline void Serialize(Stream& s, int64_t a ) { ser_writedata64(s, a); }
template<typename Stream> inline void Serialize(Stream& s, uint64_t a) { ser_writedata64(s, a); }
template<typename Stream> inline void Serialize(Stream& s, float a   ) { ser_writedata32(s, ser_float_to_uint32(a)); }
template<typename Stream> inline void Serialize(Stream& s, double a  ) { ser_writedata64(s, ser_double_to_uint64(a)); }

template<typename Stream> inline void Unserialize(Stream& s, char& a    ) { a = ser_readdata8(s); } // TODO Get rid of bare char
template<typename Stream> inline void Unserialize(Stream& s, int8_t& a  ) { a = ser_readdata8(s); }
template<typename Stream> inline void Unserialize(Stream& s, uint8_t& a ) { a = ser_readdata8(s); }
template<typename Stream> inline void Unserialize(Stream& s, int16_t& a ) { a = ser_readdata16(s); }
template<typename Stream> inline void Unserialize(Stream& s, uint16_t& a) { a = ser_readdata16(s); }
template<typename Stream> inline void Unserialize(Stream& s, int32_t& a ) { a = ser_readdata32(s); }
template<typename Stream> inline void Unserialize(Stream& s, uint32_t& a) { a = ser_readdata32(s); }
template<typename Stream> inline void Unserialize(Stream& s, int64_t& a ) { a = ser_readdata64(s); }
template<typename Stream> inline void Unserialize(Stream& s, uint64_t& a) { a = ser_readdata64(s); }
template<typename Stream> inline void Unserialize(Stream& s, float& a   ) { a = ser_uint32_to_float(ser_readdata32(s)); }
template<typename Stream> inline void Unserialize(Stream& s, double& a  ) { a = ser_uint64_to_double(ser_readdata64(s)); }

/**
 *  string
 */
template<typename Stream, typename C>
void Serialize(Stream& os, const std::basic_string<C>& str)
{
    WriteCompactSize(os, str.size());
    if (!str.empty())
        os.write((char*)&str[0], str.size() * sizeof(str[0]));
}

template<typename Stream, typename C>
void Unserialize(Stream& is, std::basic_string<C>& str)
{
    const uint64_t nSize = ReadCompactSize(is);
    str.resize(nSize);
    if (nSize != 0)
        is.read((char*)&str[0], nSize * sizeof(str[0]));
}

template<typename Stream, typename C>
void Serialize_Checked(Stream& os, const std::basic_string<C>& str, const size_t nMaxSize)
{
    if (str.size() > nMaxSize)
        throw std::ios_base::failure(strprintf("string size %zu exceeds limit %zu chars", str.size(), nMaxSize));
    Serialize(os, str);
}

template<typename Stream, typename C>
void Unserialize_Checked(Stream& is, std::basic_string<C>& str, const size_t nMaxSize)
{
    const uint64_t nStrSize = ReadCompactSize(is);
    // Limit size per read so bogus size value won't cause out of memory
    if (nStrSize > nMaxSize)
        throw std::ios_base::failure(strprintf("string size %" PRIu64 " exceeds limit %zu chars", nStrSize, nMaxSize));
    str.resize(nStrSize);
    if (nStrSize != 0)
        is.read((char*)&str[0], nStrSize * sizeof(str[0]));
}

/**
 * vector
 */
template<typename Stream, typename T, typename A>
void Serialize_impl(Stream& os, const std::vector<T, A>& v, const unsigned char&)
{
    WriteCompactSize(os, v.size());
    if (!v.empty())
        os.write((char*)&v[0], v.size() * sizeof(T));
}

template<typename Stream, typename T, typename A>
void Serialize_impl(Stream& os, const std::vector<T, A>& v, const std::shared_ptr<T>&)
{
    WriteCompactSize(os, v.size());
    for (typename std::vector<T, A>::const_iterator vi = v.begin(); vi != v.end(); ++vi)
        ::Serialize(os, (*vi));
}

template<typename Stream, typename T, typename A, typename V>
void Serialize_impl(Stream& os, const std::vector<T, A>& v, const V&)
{
    WriteCompactSize(os, v.size());
    for (typename std::vector<T, A>::const_iterator vi = v.begin(); vi != v.end(); ++vi)
        ::Serialize(os, (*vi));
}

template<typename Stream, typename T, typename A>
inline void Serialize(Stream& os, const std::vector<T, A>& v)
{
    Serialize_impl(os, v, T());
}

template<typename Stream, typename T, typename A, typename V>
void Serialize_Checked_impl(Stream& os, const std::vector<T, A>& v, const V&, const size_t nMaxSize)
{
    if (v.size() > nMaxSize)
		throw std::ios_base::failure(strprintf("vector size %zu exceeds limit %zu elements", v.size(), nMaxSize));  
    Serialize_impl(os, v, V()); 
}

template<typename Stream, typename T, typename A>
inline void Serialize_Checked(Stream& os, const std::vector<T, A>& v, const size_t nMaxSize)
{
    Serialize_Checked_impl(os, v, T(), nMaxSize);
}

template<typename Stream, typename T, typename A>
void Unserialize_impl(Stream& is, std::vector<T, A>& v, const unsigned char&)
{
    // Limit size per read so bogus size value won't cause out of memory
    v.clear();
    const uint64_t nSize = ReadCompactSize(is);
    uint64_t i = 0;
    while (i < nSize)
    {
        uint64_t blk = std::min<uint64_t>(nSize - i, 1 + 4999999 / sizeof(T));
        v.resize(i + blk);
        is.read((char*)&v[i], blk * sizeof(T));
        i += blk;
    }
}

template<typename Stream, typename T, typename A, typename V>
void Unserialize_impl(Stream& is, std::vector<T, A>& v, const V&)
{
    v.clear();
    const uint64_t nSize = ReadCompactSize(is);
    uint64_t i = 0;
    uint64_t nMid = 0;
    while (nMid < nSize)
    {
        nMid += 5000000 / sizeof(T);
        if (nMid > nSize)
            nMid = nSize;
        v.resize(nMid);
        for (; i < nMid; i++)
            Unserialize(is, v[i]);
    }
}

template<typename Stream, typename T, typename A>
inline void Unserialize(Stream& is, std::vector<T, A>& v)
{
    Unserialize_impl(is, v, T());
}

template<typename Stream, typename T, typename A, typename V>
void Unserialize_Checked_impl(Stream& is, std::vector<T, A>& v, const V&, const size_t nMaxSize)
{
    v.clear();
    const uint64_t nSize = ReadCompactSize(is);
    if (nSize > nMaxSize)
		throw std::ios_base::failure(strprintf("vector size %" PRIu64 " exceeds limit %zu elements", nSize, nMaxSize));
    uint64_t i = 0;
    uint64_t nMid = 0;
    while (nMid < nSize)
    {
        nMid += 5000000 / sizeof(T);
        if (nMid > nSize)
            nMid = nSize;
        v.resize(nMid);
        for (; i < nMid; i++)
            Unserialize(is, v[i]);
    }
}

template<typename Stream, typename T, typename A>
inline void Unserialize_Checked(Stream& is, std::vector<T, A>& v, const size_t nMaxSize)
{
    Unserialize_Checked_impl(is, v, T(), nMaxSize);
}

/**
 * If none of the specialized versions above matched, default to calling member function.
 */
template<typename Stream, typename T>
inline void Serialize(Stream& os, const T& a)
{
    a.Serialize(os);
}

template<typename Stream, typename T>
inline void Unserialize(Stream& is, T& a)
{
    a.Unserialize(is);
}

/**
 * Support for ADD_SERIALIZE_METHODS and READWRITE macro
 */
enum class SERIALIZE_ACTION
{
    NoAction = 0,
    Read = 1,
    Write = 2
};

template <typename Stream, typename _T>
inline void SerReadWrite(Stream& s, _T& obj, const SERIALIZE_ACTION ser_action)
{
    switch (ser_action)
    {
    case SERIALIZE_ACTION::Read:
        ::Unserialize(s, obj);
        break;

    case SERIALIZE_ACTION::Write:
        ::Serialize(s, obj);
        break;

    default:
        break;
    }
}

template <typename Stream, typename _T>
inline void SerReadWriteChecked(Stream& s, _T& obj, const SERIALIZE_ACTION ser_action, const size_t nMaxSize)
{
    switch (ser_action)
    {
    case SERIALIZE_ACTION::Read:
        ::Unserialize_Checked(s, obj, nMaxSize);
        break;

    case SERIALIZE_ACTION::Write:
        ::Serialize_Checked(s, obj, nMaxSize);
        break;

    default:
        break;
    }
}

/* ::GetSerializeSize implementations
 *
 * Computing the serialized size of objects is done through a special stream
 * object of type CSizeComputer, which only records the number of bytes written
 * to it.
 *
 * If your Serialize or SerializationOp method has non-trivial overhead for
 * serialization, it may be worthwhile to implement a specialized version for
 * CSizeComputer, which uses the s.seek() method to record bytes that would
 * be written instead.
 */
class CSizeComputer
{
protected:
    size_t m_nSize;

    const int m_nType;
    const int m_nVersion;

public:
    CSizeComputer(const int nTypeIn, const int nVersionIn) : 
        m_nSize(0), 
        m_nType(nTypeIn), 
        m_nVersion(nVersionIn) {}

    void write(const char *psz, const size_t nSize)
    {
        m_nSize += nSize;
    }
    void read(char* pch, const size_t nSize) {} // stub, class is used only in write-mode
    void ignore(const size_t nSizeToSkip) {}    // stub

    /** Pretend _nSize bytes are written, without specifying them. */
    void seek(const size_t nSize)
    {
        m_nSize += nSize;
    }

    template<typename T>
    CSizeComputer& operator<<(const T& obj)
    {
        ::Serialize(*this, obj);
        return (*this);
    }
    template <typename T>
    CSizeComputer& operator>>(T& obj)
    {
        return (*this);
    }

    size_t size() const noexcept { return m_nSize; }
    int GetVersion() const noexcept { return m_nVersion; }
    int GetType() const noexcept { return m_nType; }
};
