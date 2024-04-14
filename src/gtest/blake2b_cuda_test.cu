// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <tuple>

#include <gtest/gtest.h>
#include <tinyformat.h>

#include <local_types.h>
#include <blake2b.h>
#include <src/equihash/blake2b_host.h>
#include <src/kernel/blake2b_device.h>
#include <src/utils/strencodings.h>

using namespace std;
using namespace testing;

__global__ void blake2b_hash_kernel(blake2b_state* state, const uint8_t* input, size_t inlen, uint8_t* output)
{
    // Update the state with input data
    blake2b_update_device(state, input, inlen);

    // Finalize the hash and store the result in output
    blake2b_final_device(state, output, BLAKE2B_OUTBYTES);
}


class Blake2bCudaTest : public TestWithParam<tuple<string, string>>
{
protected:
    void SetUp() override
    {
        // Initialize CUDA
        cudaSetDevice(0);
    }

    void TearDown() override
    {
        // Reset CUDA device
        cudaDeviceReset();
    }    
};

TEST_P(Blake2bCudaTest, ComputesCorrectHash)
{
    // Get the input data and expected output from the test parameters
    string input = get<0>(GetParam());
    string expectedOutput = get<1>(GetParam());

    // Prepare input data
    v_uint8 vInput = ParseHex(input);
    v_uint8 vExpectedOutput = ParseHex(expectedOutput);

    // Initialize Blake2b state on the host
    blake2b_state state;
    blake2b_init_host(&state, BLAKE2B_OUTBYTES);

    // Allocate memory on the device
    blake2b_state* d_state;
    uint8_t* d_output;
    cudaMalloc(&d_state, sizeof(blake2b_state));
    cudaMalloc(&d_output, BLAKE2B_OUTBYTES);

    // Copy Blake2b state from host to device
    cudaMemcpy(d_state, &state, sizeof(blake2b_state), cudaMemcpyHostToDevice);
    uint8_t *devInput = nullptr;
    cudaMalloc(&devInput, vInput.size());
    cudaMemcpy(devInput, vInput.data(), vInput.size(), cudaMemcpyHostToDevice);

    // Launch kernel to perform Blake2b hash computation on the device
    blake2b_hash_kernel<<<1, 1>>>(d_state, devInput, vInput.size(), d_output);

    // Copy the computed hash from device to host
    uint8_t output[BLAKE2B_OUTBYTES];
    cudaMemcpy(output, d_output, BLAKE2B_OUTBYTES, cudaMemcpyDeviceToHost);

    // Compare the computed output with the expected output
    ASSERT_EQ(memcmp(output, vExpectedOutput.data(), BLAKE2B_OUTBYTES), 0) <<
        strprintf("Expected vs output:\n%s\n%s", HexStr(vExpectedOutput), HexStr(v_uint8(output, output + BLAKE2B_OUTBYTES)));

    // Free device memory
    cudaFree(d_state);
    cudaFree(d_output);
}

INSTANTIATE_TEST_SUITE_P(Blake2B, Blake2bCudaTest, 
    Values(
        make_tuple(
            "",
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce"
        ),
        make_tuple(
            "00010203",
            "77ddf4b14425eb3d053c1e84e3469d92c4cd910ed20f92035e0c99d8a7a86cecaf69f9663c20a7aa230bc82f60d22fb4a00b09d3eb8fc65ef547fe63c8d3ddce"
        ),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f9091929394",
            "abcb61cb3683d18f27ad527908ed2d32a0426cb7bb4bf18061903a7dc42e7e76f982382304d18af8c80d91dd58dd47af76f8e2c36e28af2476b4bccf82e89fdf"
        )
    )
);

__global__ void generate_hash_kernel(blake2b_state* state, uint32_t leb, uint8_t* hash)
{
    blake2b_state localState = *state;
    blake2b_update_device(&localState, reinterpret_cast<const uint8_t*>(&leb), sizeof(leb));
    blake2b_final_device(&localState, hash, BLAKE2B_OUTBYTES);
}

TEST(EquihashTest, GenerateHashHostAndKernel)
{
    // Set up the test parameters
    const uint32_t leb = 123;

    // Initialize the Blake2b state on the host
    blake2b_state hostState;
    blake2b_init_host(&hostState, BLAKE2B_OUTBYTES);

    // Allocate memory on the device for the Blake2b state and hash
    blake2b_state* d_state;
    uint8_t* d_hash;
    cudaMalloc(&d_state, sizeof(blake2b_state));
    cudaMalloc(&d_hash, BLAKE2B_OUTBYTES);

    // Copy the initialized Blake2b state from host to device
    cudaMemcpy(d_state, &hostState, sizeof(blake2b_state), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to generate the hash
    generate_hash_kernel<<<1, 1>>>(d_state, leb, d_hash);

    // Copy the generated hash from device to host
    uint8_t hostHash[BLAKE2B_OUTBYTES];
    cudaMemcpy(hostHash, d_hash, BLAKE2B_OUTBYTES, cudaMemcpyDeviceToHost);

    // Generate the hash on the host
    uint8_t expectedHash[BLAKE2B_OUTBYTES];
    blake2b_state state = hostState;
    blake2b_update_host(&state, reinterpret_cast<const uint8_t*>(&leb), sizeof(leb));
    blake2b_final_host(&state, expectedHash, BLAKE2B_OUTBYTES);

    // Compare the hashes generated by the host and kernel versions
    ASSERT_EQ(memcmp(hostHash, expectedHash, BLAKE2B_OUTBYTES), 0) <<
        "Hash mismatch";

    // Free device memory
    cudaFree(d_state);
    cudaFree(d_hash);
}

