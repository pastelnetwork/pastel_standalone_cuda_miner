// Copyright (c) 2018-2024 The Pastel Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

#if (defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__DragonFly__))
#include <pthread.h>
#include <pthread_np.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#elif defined(WIN32)
#include <fcntl.h>
#endif

#ifndef WIN32
// for posix_fallocate
#ifdef __linux__

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#define _POSIX_C_SOURCE 200112L

#endif // __linux__

#include <algorithm>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>

#else

#ifdef _MSC_VER
#pragma warning(disable : 4786 4804 4805 4717)
#include <processthreadsapi.h>
#endif

#ifdef _WIN32_WINNT
#undef _WIN32_WINNT
#endif
#define _WIN32_WINNT 0x0601

#ifdef _WIN32_IE
#undef _WIN32_IE
#endif
#define _WIN32_IE 0x0601

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#define popen _popen
#define pclose _pclose

#include <io.h> /* for _commit */
#include <shlobj.h>
#endif // WIN32

#ifdef HAVE_SYS_PRCTL_H
#include <sys/prctl.h>
#endif

void RenameThread(const char* szThreadName, void* pThreadNativeHandle)
{
    if (!szThreadName || !*szThreadName)
        return;
#if defined(PR_SET_NAME)
    // Only the first 15 characters are used (16 - NUL terminator)
    ::prctl(PR_SET_NAME, szThreadName, 0, 0, 0);
#elif (defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__DragonFly__))
    pthread_set_name_np(pthread_self(), szThreadName);

#elif defined(MAC_OSX)
    pthread_setname_np(szThreadName);
#elif defined(_MSC_VER)
    wstring wsName;
    const int nNameLen = static_cast<int>(strlen(szThreadName));
    int nRes = MultiByteToWideChar(CP_UTF8, 0, szThreadName, nNameLen, nullptr, 0);
    if (nRes > 0)
    {
        wsName.resize(nRes + 10);
        nRes = MultiByteToWideChar(CP_UTF8, 0, szThreadName, nNameLen, wsName.data(), static_cast<int>(wsName.size()));
        if (nRes > 0)
            SetThreadDescription(static_cast<HANDLE>(pThreadNativeHandle), wsName.c_str());
    }
#else
    // Prevent warnings for unused parameters...
    (void)szThreadName;
#endif
}
