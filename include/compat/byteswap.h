// Copyright (c) 2014 The Bitcoin developers
// Distributed under the MIT software license, see the accompanying
#pragma once

#ifdef HAVE_CONFIG_H
#include <compat/pastel-config.h>
#endif

#include <cstdint>

#if defined(HAVE_BYTESWAP_H)
#include <byteswap.h>
#endif

#ifndef HAVE_BYTESWAP_H

#if HAVE_DECL_BSWAP_16 == 0
inline uint16_t bswap_16(uint16_t x)
{
    return (x >> 8) | ((x & 0x00ff) << 8);
}
#endif // HAVE_DECL_BSWAP16

#if HAVE_DECL_BSWAP_32 == 0
inline uint32_t bswap_32(uint32_t x)
{
    return (((x & 0xff000000U) >> 24) | ((x & 0x00ff0000U) >>  8) |
            ((x & 0x0000ff00U) <<  8) | ((x & 0x000000ffU) << 24));
}
#endif // HAVE_DECL_BSWAP32

#if HAVE_DECL_BSWAP_64 == 0
inline uint64_t bswap_64(uint64_t x)
{
     return (((x & 0xff00000000000000ull) >> 56)
          | ((x & 0x00ff000000000000ull) >> 40)
          | ((x & 0x0000ff0000000000ull) >> 24)
          | ((x & 0x000000ff00000000ull) >> 8)
          | ((x & 0x00000000ff000000ull) << 8)
          | ((x & 0x0000000000ff0000ull) << 24)
          | ((x & 0x000000000000ff00ull) << 40)
          | ((x & 0x00000000000000ffull) << 56));
}
#endif // HAVE_DECL_BSWAP64
#endif // HAVE_BYTESWAP_H