// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <string>
#include <memory>
#include <iostream>

#include <src/utils/logger.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace std;
using namespace spdlog;

shared_ptr<logger> gl_logger;
shared_ptr<logger> gl_console_logger;


bool InitializeLogger(const string &sLogFilePath, level::level_enum log_level)
{
	try
	{
		// initialize console logger
		gl_console_logger = make_shared<logger>("console", make_shared<sinks::stdout_color_sink_mt>());
		gl_console_logger->set_level(log_level);
		gl_console_logger->set_pattern("%H:%M:%S [%^%l%$] %v");
		register_logger(gl_console_logger);

		gl_logger = make_shared<logger>("file_logger", make_shared<sinks::basic_file_sink_mt>(sLogFilePath, true));
		gl_logger->set_level(log_level);
		gl_logger->set_pattern("%H:%M:%S [%l] %v");
		register_logger(gl_logger);

		set_default_logger(gl_logger);
		set_level(log_level);
		info("Logging initialized. Log file: {}", sLogFilePath);
	} catch (const spdlog::spdlog_ex& ex)
	{
		cerr << "Log initialization failed: " << ex.what() << endl;
		return false;
	}
	return true;
}

void FinalizeLogger()
{
    if (gl_logger)
        gl_logger->info("Finalizing logging and shutting down...");
   
    // Shutdown spdlog to flush and close all loggers
    spdlog::shutdown();
    gl_logger.reset();
    gl_console_logger.reset();

	spdlog::drop("console");
	spdlog::drop("file_logger");
}
