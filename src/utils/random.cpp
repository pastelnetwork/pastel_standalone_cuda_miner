// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#include <random>

#include <src/utils/random.h>

using namespace std;

// Returns a reference to a thread-local random number generator.
// The generator is seeded upon the first use in each thread.
static mt19937_64 &random_engine()
{
    static thread_local random_device rd;
    static thread_local mt19937_64 generator(rd());
    return generator;
}

// Generates a random uint32_t value.
uint32_t random_uint32(const uint32_t nMaxValue)
{
    auto &gen = random_engine();
    uniform_int_distribution<uint32_t> dist(0, nMaxValue);
    return dist(gen);
}