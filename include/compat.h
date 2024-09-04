#pragma once
// Copyright (c) 2018-2024 The Pastel Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.

#ifdef WIN32
#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0601
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifdef FD_SETSIZE
#undef FD_SETSIZE // prevent redefinition compiler warning
#endif
#define FD_SETSIZE 1024 // max number of fds in fd_set

#include <winsock2.h>     // Must be included before mswsock.h and windows.h

#include <mswsock.h>
#include <windows.h>
#include <ws2tcpip.h>
#include <sys/types.h>

#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/resource.h>
#endif

#ifndef WIN32
// PRIO_MAX is not defined on Solaris
#ifndef PRIO_MAX
#define PRIO_MAX (20)
#endif
#ifndef PRIO_MIN
#define PRIO_MIN (-20)
#endif
#define THREAD_PRIORITY_LOWEST          PRIO_MIN
#define THREAD_PRIORITY_BELOW_NORMAL    2
#define THREAD_PRIORITY_NORMAL          0
#define THREAD_PRIORITY_ABOVE_NORMAL    (-2)
#define THREAD_PRIORITY_HIGHEST         PRIO_MAX
#endif
