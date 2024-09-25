#pragma once
// Copyright (c) 2024 The Pastel Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

/* define if the compiler supports basic C++20 syntax */
#ifndef HAVE_CXX20
#define HAVE_CXX20 1
#endif

#ifdef __linux__
/* Define to 1 if you have the <endian.h> header file. */
#define HAVE_ENDIAN_H 1
#define ENDIAN_DECL 1

/* Define to 1 if you have the <byteswap.h> header file. */
#define HAVE_BYTESWAP_H 1
#endif // __linux__

#ifdef WIN32
#define ENDIAN_DECL 0
#endif

/* Define to 1 if you have the declaration of `be16toh', and to 0 if you
   don't. */
#define HAVE_DECL_BE16TOH ENDIAN_DECL

/* Define to 1 if you have the declaration of `be32toh', and to 0 if you
   don't. */
#define HAVE_DECL_BE32TOH ENDIAN_DECL

/* Define to 1 if you have the declaration of `be64toh', and to 0 if you
   don't. */
#define HAVE_DECL_BE64TOH ENDIAN_DECL

/* Define to 1 if you have the declaration of `bswap_16', and to 0 if you
   don't. */
#define HAVE_DECL_BSWAP_16 ENDIAN_DECL

/* Define to 1 if you have the declaration of `bswap_32', and to 0 if you
   don't. */
#define HAVE_DECL_BSWAP_32 ENDIAN_DECL

/* Define to 1 if you have the declaration of `bswap_64', and to 0 if you
   don't. */
#define HAVE_DECL_BSWAP_64 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htobe16', and to 0 if you
   don't. */
#define HAVE_DECL_HTOBE16 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htobe32', and to 0 if you
   don't. */
#define HAVE_DECL_HTOBE32 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htobe64', and to 0 if you
   don't. */
#define HAVE_DECL_HTOBE64 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htole16', and to 0 if you
   don't. */
#define HAVE_DECL_HTOLE16 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htole32', and to 0 if you
   don't. */
#define HAVE_DECL_HTOLE32 ENDIAN_DECL

/* Define to 1 if you have the declaration of `htole64', and to 0 if you
   don't. */
#define HAVE_DECL_HTOLE64 ENDIAN_DECL

/* Define to 1 if you have the declaration of `le16toh', and to 0 if you
   don't. */
#define HAVE_DECL_LE16TOH ENDIAN_DECL

/* Define to 1 if you have the declaration of `le32toh', and to 0 if you
   don't. */
#define HAVE_DECL_LE32TOH ENDIAN_DECL

/* Define to 1 if you have the declaration of `le64toh', and to 0 if you
   don't. */
#define HAVE_DECL_LE64TOH ENDIAN_DECL
