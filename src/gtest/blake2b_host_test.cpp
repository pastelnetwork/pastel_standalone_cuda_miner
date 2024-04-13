// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <iostream>
#include <tuple>

#include <gtest/gtest.h>

#include <blake2b.h>
#include <tinyformat.h>
#include <local_types.h>
#include <src/utils/strencodings.h>
#include <src/equihash/blake2b_host.h>

using namespace std;
using namespace testing;

class Blake2bHostTest : public TestWithParam<tuple<string, string, string, string, string>>
{
public:
    void SetUp() override
    {
        auto params = GetParam();
        vInput = ParseHex(get<0>(params));
        vExpectedOutput = ParseHex(get<1>(params));
        vKey = ParseHex(get<2>(params));
        vSalt = ParseHex(get<3>(params));
        vPersonal = ParseHex(get<4>(params));

        if (vKey.empty())
            blake2b_init_host(&state, BLAKE2B_OUTBYTES);
        else
            blake2b_init_salt_personal_host(&state, vKey.data(), vKey.size(), BLAKE2B_OUTBYTES, 
                vSalt.data(), vSalt.size(), vPersonal.data(), vPersonal.size());
    }

protected:
    blake2b_state state;
    v_uint8 vInput;
    v_uint8 vExpectedOutput;
    v_uint8 vKey;
    v_uint8 vSalt;
    v_uint8 vPersonal;
}; 

TEST_P(Blake2bHostTest, ComputesCorrectHash)
{
    uint8_t output[BLAKE2B_OUTBYTES];

    // Update the state with input data
    blake2b_update_host(&state, vInput.data(), vInput.size());

    // Finalize the hash and store the result in output
    blake2b_final_host(&state, output, BLAKE2B_OUTBYTES);

    // Compare the computed output with the expected output
    ASSERT_EQ(memcmp(output, vExpectedOutput.data(), BLAKE2B_OUTBYTES), 0) <<
        strprintf("Expected vs output:\n%s\n%s", HexStr(vExpectedOutput), HexStr(v_uint8(output, output + BLAKE2B_OUTBYTES)));
}

INSTANTIATE_TEST_SUITE_P(Blake2B, Blake2bHostTest,
    Values(
        make_tuple(
            "",
            "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
            "", "", ""),
        make_tuple(
            "00010203",
            "77ddf4b14425eb3d053c1e84e3469d92c4cd910ed20f92035e0c99d8a7a86cecaf69f9663c20a7aa230bc82f60d22fb4a00b09d3eb8fc65ef547fe63c8d3ddce",
            "", "", ""),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728",
            "25261eb296971d6e4a71b2928e64839c67d422872bf9f3c31993615222de9f8f0b2c4be8548559b4b354e736416e3218d4e8a1e219a4a6d43e1a9a521d0e75fc",
            "", "", ""),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728",
            "25261eb296971d6e4a71b2928e64839c67d422872bf9f3c31993615222de9f8f0b2c4be8548559b4b354e736416e3218d4e8a1e219a4a6d43e1a9a521d0e75fc",
            "", "", ""),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
            "2fc6e69fa26a89a5ed269092cb9b2a449a4409a7a44011eecad13d7c4b0456602d402fa5844f1a7a758136ce3d5d8d0e8b86921ffff4f692dd95bdc8e5ff0052",
            "", "", ""),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f808182838485868788898a8b8c8d8e8f9091929394",
            "abcb61cb3683d18f27ad527908ed2d32a0426cb7bb4bf18061903a7dc42e7e76f982382304d18af8c80d91dd58dd47af76f8e2c36e28af2476b4bccf82e89fdf",
            "", "", ""),
        // key
        make_tuple(
            "000102030405060708090a0b0c0d0e0f10111213141516171819",
            "f0d2805afbb91f743951351a6d024f9353a23c7ce1fc2b051b3a8b968c233f46f50f806ecb1568ffaa0b60661e334b21dde04f8fa155ac740eeb42e20b60d764",
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
            "", ""),
        make_tuple(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f404142434445464748494a4b4c4d4e4f505152535455565758595a5b5c5d5e5f606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f80818283",
            "a7803bcb71bc1d0f4383dde1e0612e04f872b715ad30815c2249cf34abb8b024915cb2fc9f4e7cc4c8cfd45be2d5a91eab0941c7d270e2da4ca4a9f7ac68663a",
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f",
            "", "")
    )
);

