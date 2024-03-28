// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once

#include <cstring>
#include <cassert>
#include <ios>

#include <local_types.h>
#include <src/utils/serialize.h>

// Byte-vector that clears its contents before deletion.
typedef std::vector<char> CSerializeData;

/** Double ended buffer combining vector and stream-like interfaces.
 *
 * >> and << read and write unformatted data using the above serialization templates.
 * Fills with data in linear time; some stringstream implementations take N^2 time.
 */
template<typename SerializeType>
class CBaseDataStream
{
protected:
    typedef SerializeType vector_type;
    vector_type vch;
    size_t nReadPos;

    int m_nType;
    int m_nVersion;

public:

    typedef typename vector_type::allocator_type   allocator_type;
    typedef typename vector_type::size_type        size_type;
    typedef typename vector_type::difference_type  difference_type;
    typedef typename vector_type::reference        reference;
    typedef typename vector_type::const_reference  const_reference;
    typedef typename vector_type::value_type       value_type;
    typedef typename vector_type::iterator         iterator;
    typedef typename vector_type::const_iterator   const_iterator;
    typedef typename vector_type::reverse_iterator reverse_iterator;

    explicit CBaseDataStream(const int nType, const int nVersion)
    {
        Init(nType, nVersion);
    }

    CBaseDataStream(CBaseDataStream&& p) noexcept :
        vch(move(p.vch))
    {
        Init(p.m_nType, p.m_nVersion);
        nReadPos = p.nReadPos;
        p.nReadPos = 0;
    }

    CBaseDataStream& operator=(CBaseDataStream&& p) noexcept
    {
        if (this != &p)
        {
            vch = move(p.vch);
            Init(p.m_nType, p.m_nVersion);
            nReadPos = p.nReadPos;
            p.nReadPos = 0;
        }
        return (*this);
    }

    CBaseDataStream(const CBaseDataStream& p) :
        vch(p.vch)
    {
		Init(p.m_nType, p.m_nVersion);
		nReadPos = p.nReadPos;
	}

    CBaseDataStream& operator=(const CBaseDataStream& p)
    {
        if (this != &p)
        {
			vch = p.vch;
			Init(p.m_nType, p.m_nVersion);
			nReadPos = p.nReadPos;
		}
		return (*this);
	}

    CBaseDataStream(const_iterator pbegin, const_iterator pend, const int nType, const int nVersion) : 
        vch(pbegin, pend)
    {
        Init(nType, nVersion);
    }

#if !defined(_MSC_VER) || _MSC_VER >= 1300
    CBaseDataStream(const char* pbegin, const char* pend, const int nType, const int nVersion) : 
        vch(pbegin, pend)
    {
        Init(nType, nVersion);
    }
#endif

    CBaseDataStream(vector_type&& vchIn, const int nType, const int nVersion) : 
        vch(move(vchIn))
    {
        Init(nType, nVersion);
    }

    CBaseDataStream(const vector_type& vchIn, const int nType, const int nVersion) : 
        vch(vchIn.cbegin(), vchIn.cend())
    {
        Init(nType, nVersion);
    }

    CBaseDataStream(const v_uint8& vchIn, const int nType, const int nVersion) : 
        vch(vchIn.cbegin(), vchIn.cend())
    {
        Init(nType, nVersion);
    }

    void Init(const int nType, const int nVersion)
    {
        nReadPos = 0;
        m_nType = nType;
        m_nVersion = nVersion;
    }

    CBaseDataStream& operator+=(const CBaseDataStream& b)
    {
        vch.insert(vch.end(), b.cbegin(), b.cend());
        return *this;
    }

    friend CBaseDataStream operator+(const CBaseDataStream& a, const CBaseDataStream& b)
    {
        CBaseDataStream ret = a;
        ret += b;
        return (ret);
    }

    std::string str() const
    {
        return (std::string(cbegin(), cend()));
    }

    //
    // Vector subset
    //
    const_iterator begin() const                     { return vch.begin() + nReadPos; }
    const_iterator cbegin() const                    { return vch.cbegin() + nReadPos; }
    iterator begin() { return vch.begin() + nReadPos; }
    const_iterator end() const                       { return vch.end(); }
    const_iterator cend() const                      { return vch.cend(); }
    iterator end() { return vch.end(); }
    size_type size() const noexcept                  { return vch.size() - nReadPos; }
    bool empty() const noexcept                      { return vch.size() == nReadPos; }
    void resize(size_type n, value_type c = 0)         { vch.resize(n + nReadPos, c); }
    void reserve(size_type n)                        { vch.reserve(n + nReadPos); }
    const_reference operator[](size_type pos) const  { return vch[pos + nReadPos]; }
    reference operator[](size_type pos)              { return vch[pos + nReadPos]; }
    void clear() noexcept                            { vch.clear(); nReadPos = 0; }
    iterator insert(iterator it, const char& x=char()) { return vch.insert(it, x); }
    void insert(iterator it, size_type n, const char& x) { vch.insert(it, n, x); }
    size_t getReadPos() const noexcept               { return nReadPos; }
    void extractData(v_uint8 &v)
    {
        v.resize(vch.size());
        memcpy(v.data(), vch.data(), vch.size());
    }

    void insert(iterator it, std::vector<char>::const_iterator first, std::vector<char>::const_iterator last)
    {
        if (last == first)
            return;
        assert(last - first > 0);
        if (it == vch.begin() + nReadPos && (unsigned int)(last - first) <= nReadPos)
        {
            // special case for inserting at the front when there's room
            nReadPos -= (last - first);
            memcpy(&vch[nReadPos], &first[0], last - first);
        }
        else
            vch.insert(it, first, last);
    }

#if !defined(_MSC_VER) || _MSC_VER >= 1300
    void insert(iterator it, const char* first, const char* last)
    {
        if (last == first)
            return;
        assert(last - first > 0);
        if (it == vch.begin() + nReadPos && (unsigned int)(last - first) <= nReadPos)
        {
            // special case for inserting at the front when there's room
            nReadPos -= (last - first);
            memcpy(&vch[nReadPos], &first[0], last - first);
        }
        else
            vch.insert(it, first, last);
    }
#endif

    iterator erase(iterator it)
    {
        if (it == vch.begin() + nReadPos)
        {
            // special case for erasing from the front
            if (++nReadPos >= vch.size())
            {
                // whenever we reach the end, we take the opportunity to clear the buffer
                nReadPos = 0;
                return vch.erase(vch.begin(), vch.end());
            }
            return vch.begin() + nReadPos;
        }
        else
            return vch.erase(it);
    }

    iterator erase(iterator first, iterator last)
    {
        if (first == vch.begin() + nReadPos)
        {
            // special case for erasing from the front
            if (last == vch.end())
            {
                nReadPos = 0;
                return vch.erase(vch.begin(), vch.end());
            }
            else
            {
                nReadPos = (last - vch.begin());
                return last;
            }
        }
        else
            return vch.erase(first, last);
    }

    inline void Compact()
    {
        vch.erase(vch.begin(), vch.begin() + nReadPos);
        nReadPos = 0;
    }

    bool rewind(size_type n)
    {
        // rewind by n characters if the buffer hasn't been compacted yet
        if (n > nReadPos)
            return false;
        nReadPos -= n;
        return true;
    }

    //
    // Stream subset
    //
    bool eof() const noexcept     { return size() == 0; }
    CBaseDataStream* rdbuf()      { return this; }

    void SetType(const int nType) noexcept    { m_nType = nType; }
    int GetType() const noexcept          { return m_nType; }
    void SetVersion(const int nVersion) noexcept { m_nVersion = nVersion; }
    int GetVersion() const noexcept       { return m_nVersion; }

    bool operator==(const CBaseDataStream& ds) const
    {
        if (vch.size() != ds.vch.size())
            return false;
        return memcmp(vch.data(), ds.vch.data(), vch.size()) == 0;
    }

    void read(char* pch, const size_t nSize)
    {
        if (nSize == 0) 
            return;

        if (!pch)
            throw std::ios_base::failure("CBaseDataStream::read(): cannot write from null pointer");

        // Read from the beginning of the buffer
        const size_t nReadPosNext = nReadPos + nSize;
        if (nReadPosNext >= vch.size())
        {
            if (nReadPosNext > vch.size())
                throw std::ios_base::failure("CBaseDataStream::read(): end of data");
            memcpy(pch, &vch[nReadPos], nSize);
            nReadPos = 0;
            vch.clear();
            return;
        }
        memcpy(pch, &vch[nReadPos], nSize);
        nReadPos = nReadPosNext;
    }

    void read(CBaseDataStream &os, const size_t nSize)
    {
        if (nSize == 0) 
            return;

        // Read from the beginning of the buffer
        const size_t nReadPosNext = nReadPos + nSize;
        if (nReadPosNext >= vch.size())
        {
            if (nReadPosNext > vch.size())
                throw std::ios_base::failure("CBaseDataStream::read(): end of data");
            os.write(&vch[nReadPos], nSize);
            nReadPos = 0;
            vch.clear();
            return;
        }
        os.write(&vch[nReadPos], nSize);
        nReadPos = nReadPosNext;
    }

    void read_buf(uint8_t* pch, const size_t nSize) const
    {
        if (nSize == 0)
            return;

        if (!pch)
            throw std::ios_base::failure("CBaseDataStream::read(): cannot read from null pointer");

        // Read from the beginning of the buffer
        const size_t nReadPosNext = nReadPos + nSize;
        if (nReadPosNext > vch.size())
            throw std::ios_base::failure("CBaseDataStream::read(): end of data");
        memcpy(pch, &vch[nReadPos], nSize);
    }

    void ignore(const size_t nSize)
    {
        // Ignore from the beginning of the buffer
        const size_t nReadPosNext = nReadPos + nSize;
        if (nReadPosNext >= vch.size())
        {
            if (nReadPosNext > vch.size())
                throw std::ios_base::failure("CBaseDataStream::ignore(): end of data");
            nReadPos = 0;
            vch.clear();
            return;
        }
        nReadPos = nReadPosNext;
    }

    void write(const char* pch, const size_t nSize)
    {
        // Write to the end of the buffer
        vch.insert(vch.end(), pch, pch + nSize);
    }

    template<typename Stream>
    void Serialize(Stream& s) const
    {
        // Special case: stream << stream concatenates like stream += stream
        if (!vch.empty())
            s.write((char*)&vch[0], vch.size() * sizeof(vch[0]));
    }

    template<typename T>
    CBaseDataStream& operator<<(const T& obj)
    {
        // Serialize to this stream
        ::Serialize(*this, obj);
        return (*this);
    }

    template<typename T>
    CBaseDataStream& operator>>(T& obj)
    {
        // Unserialize from this stream
        ::Unserialize(*this, obj);
        return (*this);
    }

    void GetAndClear(CSerializeData &d) {
        d.insert(d.end(), begin(), end());
        clear();
    }
};

class CDataStream : public CBaseDataStream<CSerializeData>
{
public:
    explicit CDataStream(const int nType, const int nVersion) :
        CBaseDataStream(nType, nVersion)
    {}

    CDataStream(const_iterator pbegin, const_iterator pend, const int nType, const int nVersion) :
        CBaseDataStream(pbegin, pend, nType, nVersion)
    {}

#if !defined(_MSC_VER) || _MSC_VER >= 1300
    CDataStream(const char* pbegin, const char* pend, const int nType, const int nVersion) :
        CBaseDataStream(pbegin, pend, nType, nVersion)
    {}
#endif

    CDataStream(const vector_type& vchIn, const int nType, const int nVersion) :
        CBaseDataStream(vchIn, nType, nVersion)
    {}

    CDataStream(const v_uint8& vchIn, const int nType, const int nVersion) :
        CBaseDataStream(vchIn, nType, nVersion)
    {}

    template <typename... Args>
    CDataStream(const int nType, const int nVersion, Args&&... args) :
        CBaseDataStream(nType, nVersion, args...)
    {}

    template <typename T>
    CDataStream& operator<<(const T& obj)
    {
        // Serialize to this stream
        ::Serialize(*this, obj);
        return (*this);
    }

    template <typename T>
    CDataStream& operator>>(T& obj)
    {
        // Unserialize from this stream
        ::Unserialize(*this, obj);
        return (*this);
    }
};
