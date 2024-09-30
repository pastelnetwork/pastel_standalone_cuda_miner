# Pastel Standalone CUDA Miner

Pastel CUDA Miner uses cmake build system.
Some dependent projects are used (automatically downloaded and built by cmake):
 - LibEvent
 - Google Test library
 - spdlog

Windows Build:

Use presets to generate Visual Studio 2022 projects (Debug or Release):
 cmake --preset vs2022_dbg
 cmake --preset vs2022

Projects will be generated in [build-aux/vs2022/Release] or [build-aux/vs2022/Debug].
Run from that folder [cmake --build .] to build pastel_miner.
Binary will be moved to [bin/Debug] or [bin/Release] folder.

Linux Build:

Use presets to generate Linux makefiles:
 cmake --preset linux
 cmake --preset linux_dbg

Project files will be generated in [build-aux/linux/release] or [build-aux/linux/debug].
Run [cmake --build .] from these folders to build pastel_miner.


Configuration file pastel_miner.conf is required to run Pastel Miner:
```ini
# s-nomp server address, default: localhost
server=<ip_address>
# s-nomp server port, default: 3255
port=3255

# miner address
miner_address=<miner_address>.<miner_external_ip_address>

# pool authentication password
auth_password=<password>
