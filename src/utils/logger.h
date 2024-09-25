#pragma once
// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <string>

#include <spdlog/spdlog.h>

bool InitializeLogger(const std::string &sLogFilePath, spdlog::level::level_enum log_level = spdlog::level::info);
void FinalizeLogger();

extern std::shared_ptr<spdlog::logger> gl_logger;
extern std::shared_ptr<spdlog::logger> gl_console_logger;
