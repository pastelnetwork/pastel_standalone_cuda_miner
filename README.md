# Pastel Standalone CUDA Miner

The Pastel CUDA Miner uses the CMake build system.
Several dependent projects are used (automatically downloaded and built by CMake):
 - LibEvent
 - Google Test Library
 - spdlog

## Windows Build

Use the following presets to generate Visual Studio 2022 projects (Debug or Release):
```
cmake --preset vs2022_dbg
cmake --preset vs2022
```
The projects will be generated in `[build-aux/vs2022/Release]` or `[build-aux/vs2022/Debug]`.
Run the following command from that folder to build `pastel_miner`:
```
cmake --build .
```
The binary will be moved to the `[bin/Debug]` or `[bin/Release]` folder.

## Linux Build

Use the following presets to generate Linux makefiles:
```
cmake --preset linux
cmake --preset linux_dbg
```
The project files will be generated in `[build-aux/linux/release]` or `[build-aux/linux/debug]`.
Run the following command from these folders to build `pastel_miner`:
```
cmake --build .
```

## Configuration

The configuration file `pastel_miner.conf` is required to run Pastel Miner. The structure is as follows:

```ini
# s-nomp server address (default: localhost)
server=<ip_address>

# s-nomp server port (default: 3255)
port=3255

# Miner address
miner_address=<miner_address>.<miner_external_ip_address>

# Pool authentication password
auth_password=<password>
