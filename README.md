# pastel_standalone_cuda_miner

Windows Build:

- LibEvent library build and configure:
  - download and extract libevent to ./depends/libevent
  - execute:
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./bin
  cmake --build ./build --config Release --parallel
  cmake --install build

cmake --build ./build -DCMAKE_DEBUG_POSTFIX --parallel

- Google Test libraries build and configure:
  - download and extract gtest to ./depends/gtest
  - execute:
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX=./bin
  cmake --build ./build --config Release --parallel
  cmake --install build

generate cmake project for Visual Studio 2022:
cmake -S . -B build-aux -G "Visual Studio 17 2022" -DLibEvent_ROOT="./depends/libevent/bin" -DGTest_ROOT="./depends/gtest/bin"

