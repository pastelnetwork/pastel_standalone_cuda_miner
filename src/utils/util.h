#pragma once
// Copyright (c) 2018-2024 The Pastel Core developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

// rename thread
void RenameThread(const char* szThreadName, void *pThreadNativeHandle = nullptr);
bool SetupNetworking();
