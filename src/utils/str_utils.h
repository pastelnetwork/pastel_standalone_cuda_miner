// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <algorithm>

#include <local_types.h>

static constexpr size_t DEFINE_SIZE = static_cast<size_t>(-1);

/**
 * test if character is white space not using locale.
 * 
 * \param ch - character to test`
 * \return true if the character ch is a whitespace
 */
static inline bool isspaceex(const char ch)
{
    return (ch == 0x20) || (ch >= 0x09 && ch <= 0x0D);
}

/**
 * Check if character is in lowercase (a..z).
 *
 * \param c - character to check
 * \return true - if c is in lowercase
 */
static inline bool islowerex(const char c) noexcept
{
    return (c >= 'a' && c <= 'z');
}

/**
 * Check if character is in uppercase (A..Z).
 *
 * \param c - character to check
 * \return true - if c is in uppercase
 */
static inline bool isupperex(const char c) noexcept
{
    return (c >= 'A' && c <= 'Z');
}

/**
 * Check if character is alphabetic without using locale
 *
 * \param c - character to test
 * \return true if character is alphabetic (A..Z,a..z)
 */
static inline bool isalphaex(const char c) noexcept
{
    return ((c >= 'A') && (c <= 'Z')) || ((c >= 'a') && (c <= 'z'));
}

/**
 * Check if character is decimal digit without using locale
 *
 * \param c - character to test
 * \return true if character is digit (0..9)
 */
static inline bool isdigitex(const char c) noexcept
{
    return (c >= '0') && (c <= '9');
}

/**
 * Check if character is alphanumeric without using locale.
 *
 * \param c - character to test
 * \return true if c is in one of these sets (A..Z, a..z, 0..9)
 */
static inline bool isalnumex(const char c) noexcept
{
    return isalphaex(c) || isdigitex(c);
}

/**
 * trim string in-place from start (left trim).
 *
 * \param s - string to ltrim
 */
static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.cbegin(), s.cend(), [](const auto ch) { return !isspaceex(ch); }));
}

/**
 * trim string in-place from end (right trim).
 *
 * \param s - string to rtrim
 */
static inline void rtrim(std::string& s)
{
    s.erase(std::find_if(s.crbegin(), s.crend(), [](const auto ch) { return !isspaceex(ch); }).base(), s.end());
}

/**
 * trim string in-place (both left & right trim).
 */
static inline void trim(std::string& s)
{
    ltrim(s);
    rtrim(s);
}

/**
 * Split string s with delimiter chDelimiter into vector v.
 * 
 * \param v - output vector of strings
 * \param s - input string
 * \param chDelimiter  - string parts delimiter
 */
static void str_split(v_strings &v, const std::string &s, const char chDelimiter)
{
    v.clear();
    std::string::size_type posStart = 0;
    for (std::string::size_type posEnd = 0; (posEnd = s.find(chDelimiter, posEnd)) != std::string::npos; ++posEnd)
    {
        v.emplace_back(s.substr(posStart, posEnd - posStart));
        posStart = posEnd + 1;
    }
    v.emplace_back(s.substr(posStart));
}

/**
 * Returns empty sz-string in case szStr = nullptr.
 * 
 * \param szStr - input string or nullptr
 * \return non-null string
 */
static inline const char* SAFE_SZ(const char* szStr) noexcept
{
    return szStr ? szStr : "";
}
