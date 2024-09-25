// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

//To compile the code, you'll need to use the NVIDIA CUDA Compiler (nvcc) with appropriate flags. For example:
// nvcc -arch=sm_xx -o program program.cu
//Replace xx with the appropriate compute capability of your CUDA device.

#ifdef __linux__
#include <unistd.h>
#include <sys/socket.h>
#endif
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <csignal>

#include <event2/thread.h>

#include <src/equihash/block.h>
#include <src/stratum/client.h>
#include <src/utils/svc_thread.h>
#include <src/utils/util.h>
#include <src/utils/logger.h>
#include <src/utils/config-parser.h>

using namespace std;
namespace fs = std::filesystem;

#ifdef __MINGW64__
__thread CServiceThread *funcThreadObj;
#else
thread_local CServiceThread* funcThreadObj = nullptr;
#endif
atomic_bool gl_bStopMining = false;
constexpr auto POOL_RECONNECT_INTERVAL_SECS = 15s;

constexpr auto MAJOR_VERSION = 1;
constexpr auto MINOR_VERSION = 0;

// Signal handler for Ctrl-C
void signalInterruptHandler(int signum)
{
    gl_console_logger->error("\nInterrupt signal ({}) received. Stopping mining...\n", signum);
    gl_bStopMining = true;
}

// Function to get the directory of the running executable
fs::path getExecutableDir()
{
    char buffer[1024];
    string sExecDir;

#ifdef _WIN32
    // Windows: Use GetModuleFileName to get the executable path
    GetModuleFileName(nullptr, buffer, sizeof(buffer));
    sExecDir = string(buffer);
#elif __linux__
    // Linux: Read the /proc/self/exe symbolic link
    ssize_t len = readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len != -1)
    {
        buffer[len] = '\0';
        sExecDir = string(buffer);
    } else
        throw runtime_error("Failed to read the executable path");
#endif

    // Extract the directory from the full path
    return fs::path(sExecDir).parent_path();
}

// Main program
int main()
{
	InitializeLogger("pastel_miner.log", spdlog::level::info);
    SetupNetworking();

    // Register signal handler for Ctrl-C
    signal(SIGINT, signalInterruptHandler);

    gl_console_logger->info("Pastel CUDA Equihash Miner v{}.{}", MAJOR_VERSION, MINOR_VERSION);

#ifdef WIN32
    evthread_use_windows_threads();
#else
    evthread_use_pthreads();
#endif

    CConfigParser configParser;
    string error;
    auto execDir = getExecutableDir();
	string sConfigFilePath = (execDir / "pastel_miner.conf").string();
    if (!configParser.load(error, sConfigFilePath))
    {
        gl_console_logger->error(error);
		return 1;
    }

	const auto sServerHost = configParser.getOrDefault<string>("server", "localhost");
	const auto nServerPort = configParser.getOrDefault<uint16_t>("port", 3255);

	const auto sMinerAddress = configParser.get("miner_address");
    if (!sMinerAddress.has_value())
    {
		gl_console_logger->error("miner_address option is not specified in the config file");
		return 1;
    }
    const auto nMinerThreadCount = configParser.getOrDefault<uint32_t>("miner_threads", 1);
	const auto sAuthPassword = configParser.get("auth_password");
    if (!sAuthPassword.has_value())
    {
		gl_console_logger->error("auth_password option is not specified in the config file");
		return 1;
    }

    srand(time(0));

    vector<unique_ptr<CStratumClient>> vClients;
	for (size_t i = 0; i < nMinerThreadCount; ++i)
    {
        auto& client = vClients.emplace_back(make_unique<CStratumClient>());
        if (!client->initMiningThread(error))
        {
            gl_console_logger->error("Failed to initialize mining thread: {}", error);
			return 1;
        }
        client->setServerInfo(sServerHost, nServerPort);
    	client->setAuthInfo(i + 1, get<string>(sMinerAddress.value()), get<string>(sAuthPassword.value()));
        client->startHandlingLoop();
    }

    while (!gl_bStopMining)
    {
        // connect, subscribe, authorize and start mining
        for (auto& client: vClients)
            client->checkConnectionState();

        this_thread::sleep_for(POOL_RECONNECT_INTERVAL_SECS);
    }
    for (auto& client : vClients)
		client->breakHandlingLoop();
    vClients.clear();

    gl_console_logger->info("Miner stopped");
	FinalizeLogger();
    return 0;
}