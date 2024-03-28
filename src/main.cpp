// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.

//To compile the code, you'll need to use the NVIDIA CUDA Compiler (nvcc) with appropriate flags. For example:
// nvcc -arch=sm_xx -o program program.cu
//Replace xx with the appropriate compute capability of your CUDA device.

#include <iostream>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <sstream>
#include <ctime>


#include <src/equihash/block.h>
#include <src/stratum/client.h>

using namespace std;

// Main program
int main()
{
    cout << "Pastel CUDA Equihash Miner" << endl;

    CStratumClient client("144.126.137.164", 1234);
    client.setAuthInfo("Pastel-Miner", "test_pswd");

    srand(time(0));

    // connect, subscribe, authorize and start mining
    client.handlingLoop();

    return 0;
}