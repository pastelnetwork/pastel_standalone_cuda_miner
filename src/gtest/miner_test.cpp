// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <limits>

#include <gtest/gtest.h>
#include <blake2b.h>
#include <local_types.h>
#include <src/utils/uint256.h>
#include <src/utils/strencodings.h>
#include <src/utils/streams.h>
#include <src/equihash/blake2b_host.h>
#include <src/equihash/equihash.h>
#include <src/stratum/miner.h>

using namespace testing;
using namespace std;

TEST(MinerTest, CollisionBitMask)
{
    for (int round = 0; round < Eh200_9::WK; round++)
    {
        const uint32_t collisionBitLength = Eh200_9::CollisionBitLength;
        const uint32_t globalBitOffset = round * collisionBitLength;
        uint32_t wordOffset = globalBitOffset / numeric_limits<uint32_t>::digits;
        if (wordOffset >= Eh200_9::HashWords - 1)
            wordOffset = Eh200_9::HashWords - 2;
        const uint32_t bitOffset = globalBitOffset - wordOffset * numeric_limits<uint32_t>::digits;
        const uint64_t collisionBitMask = ((1ULL << collisionBitLength) - 1) << bitOffset;
        cout << "round: " << round << ", word-offset: " << dec << wordOffset << ", collisionBitMask: " << hex << setfill('0') << setw(16) << collisionBitMask << dec << endl;
    }
}

// Test case for the mining loop
TEST(MinerTest, MiningLoop)
{
    // Set up initial blake2b_state
    blake2b_state state;
   
    auto eh = EhSolver200_9();

    eh.InitializeState(state, DEFAULT_EQUIHASH_PERS_STRING);

    // devnet block 34'000
    CBlockHeader blockHeader;
    blockHeader.nVersion = CBlockHeader::VERSION_SIGNED_BLOCK;
    blockHeader.hashPrevBlock = uint256S("0004805cdf14342bb57a9e32e7edb2ebef32e495680e3fc7b975d5607a86e3d6");
    blockHeader.hashMerkleRoot = uint256S("ccaf5c83728ba4e0a05765e38b17e2073950b0f860b610d8a3c0f1865811975b");
    blockHeader.hashFinalSaplingRoot = uint256S("3e49b5f954aa9d3545bc6c37744661eea48d7c34e3000d82b7f0010c30f4c2fb");
    blockHeader.nTime = 1711574842;
    blockHeader.nBits = 0x1f3b1ad8;
    blockHeader.nNonce = uint256S("0000b76acb89f1d761183d88cef87258169d8d2bb1e8e33195c3f764ca6c010a");
    blockHeader.sPastelID = "jXXQgGTfenHGk3tcDEjnq5PT1T5xag8ERhgAF8oExmwzLakosm8TVuS6EajED1VQmDmwNtD3Pux7w813pDndDr";
    blockHeader.prevMerkleRootSignature = ParseHex("ececf6d05012684fd5dd43f43633829c48d5f16d82318df358cf2cf97daf00b243a4bbbd5ab37b9a40bd7d5e5236fdc70f1ec5c21aa9fdd1006ff4ff4c4b59da1737be69c097ade0e4a4537deed8ef955c8cdf252f241920ca50c23714bda055b31a1ae1b5ac4c4dc46003d792719dfb2200");

    // existing solution:
    // 004eee09c90858aafcd131e6f3b2f3bcf1c3cadcb1259a61a611239b8b1fedd280f574477cd6bcfd3470013bd9e6791124253f2d1058d1a3f65c6a7771712903b1098fdc6ea3a796a4a1de994815e4da2d76d9040651372ca4acd34ff3ce62adb77dc199e902969412176fe7cb0578ab6beaff2a97247b08d3823a3eab4309fd746155e7ca676c83515e3d7f27397ace10784b1ef35ac4d00ba3c538a2f6bcb4f71f8a58e6ff7f650ad88c9de870f32faa7fab3fc46d26e340f47d66161c95cfeac608f5f121106471a3a9b5f932c25c10c6141df26920916183d66ee16f5ac86378b92412c96f18f7323f3c08a16f1ad4e9178450fa92871478802b21c816e669632a8fa693525cc1c0aecd180a4a75a557252c3b87f2af61efd2d83f68e8c916c58f7a9ab1405a3d9e3f6f836b871936a985d93752818bba80907391c39cedecb955cf95179830453c46aa7a7bbdec0208e435c481546a255fc4a35eec0ec24d0cd9921d032494e50b90f6c541cbd1b8e4d0bc114250fb888a030dc0ccd942f1c85ac2412f3c409f7f8aee9f022f185e58db0012d6b94c95e3b85ee79b362f3596c0291223844d7b46e40efa83151811d1dcbd93c794bdfa1ce8b3ccd2a9a7477d5df4230fe02a05802257ef4e165726df03caacdf203291d35f65e344a0ba3b7e66450c0285322eff81d32134f1c87a07fdcc7a8fa063060f96d2838ceed7fb2a32ef61f81627242f39e7c807a2027d1f9197f1c7e3c1868623d3c14bb8ba8339100f4a6246767645e0d4d67124675e9dc708af1bcd2d2e797b96570309531b43e4f9d885a205a61d124b0bee3ee63fdaebd3984fb377217bc49ac6a9b659523c2654b19699c354e3fbb4d54fecfce5af939f560012e6ae8100977bdd8b4c115a92a1d3893c4a587449264223b47617fb5fedb1858c0838987e3b6472c0df024ef9df8828936982a6195e0fcd22269617fe56e8418c8c9ffc55b60d99c8761b8efd4881aa40790672035b67e38e89cf04735fb06d25e35d956de11ac53a0c605358359e77571d013680fc6b27369ff85ac9bc0e06b1ef2e5ab3d9e0345170cd68c58d5ddb53f13912d7e6adc716268ce85ee4fcb06aef89cd82f4faf7148c54a9f1cb8d466984224c805c8db0d5b80aafbd1690ce515a590779c35dc44e6665c06a296ed220451127635fa42bf5b7eed2033b1bc8c729815453d05a15103781a4e7040b9e9c418894fa0402fc9e5f1d4814eefa533e4c0823dbddf21ff9ab80956a8877853728d21f04b2a6a0154e63030694600ac0ffea98d2d0138897ca154ac0d32762c2e1bde1d6a9ad664fed0b38b4a61ceba7616780dd73922bc8441dbc25f7f29717f99a794ba437174e61c62f635e09bda4b9b00cc82bc6bb27195a03af4d9c7740c97f493e15d4ddbbb205e6b07f6d22232fad6d3447b66f275d13d27570bd0c9ab6b70f1c6bd3c874f1a336da140556198d9d3728439fd392202713722462ed3a4fbb2af5eb3830502f4693aa575d6371462ab5baea501355c0d39a152b0ed3fb6bff178da9c570e6954472af5adbcc9f18a415c97283a99bc4290600d6ac8252e306d31a7f4a6a2ed35af2208ec1493d07eb6194dfee1712a6dbcd8133319c5cfb24cbb15c8258c9c36e296b04627cde0309fde440d807ba2bfe27258a74f182f9a9743240eb2183de9cca97514fa1dc83c8e253a7c9fd2b16da080b7673338ec54147f44b10bc1c6b7b74b918cdcd512b0df2fae290094128db69d6d55653feae38df82eb0c75267c751eee15bb7602a94960443a9c50f583d03a5792a713b7e4afe9da245d4875161592f23bf95b2093331172f0c7a7f7b0248ec1247e52103f9cffe226aebe623199f503bed5826b0b548de639d5de26c6

    // Initialize initialState with the desired values
    CEquihashInput ehInput(blockHeader);
    CDataStream ss(SER_NETWORK, PROTOCOL_VERSION);
    ss << ehInput;
    blake2b_update_host(&state, (unsigned char*)&ss[0], ss.size());

    uint32_t nExtraNonce2 = 123;
    string sTime = "current_time";
    size_t nIterations = 1;
    uint32_t threadsPerBlock = 256;

    // Define the generateNonce function
    auto generateNonceFn = [&](uint32_t nExtraNonce2) -> const uint256
    {
        return blockHeader.nNonce;
    };

    v_strings vHexSolutions;
    // Define the submitSolution function
    auto submitSolutionFn = [&](uint32_t nExtraNonce2, const string& sTime, const string& sNonce, const string& sHexSolution)
    {
        vHexSolutions.push_back(sHexSolution);
    };

    // Call the miningLoop function
    uint32_t solutionCount = miningLoop<EhSolver200_9>(state, nExtraNonce2, sTime, nIterations, threadsPerBlock, generateNonceFn, submitSolutionFn);

    size_t i = 0;
    for (const auto& solution : vHexSolutions)
    {
        cout << "solution #" << i << ": " << solution << endl;
        i++;
    }
    // Check the solution count
    EXPECT_GT(solutionCount, 0);

    // Additional assertions or checks can be added as needed
}

int main(int argc, char **argv)
{
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}