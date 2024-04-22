// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>

#include <blake2b.h>
#include <local_types.h>
#include <src/kernel/memutils.h>

//#define USE_DEBUG_MODE
#ifdef USE_DEBUG_MODE
#define DEBUG_FN(func) func
#else
#define DEBUG_FN(func)
#endif

//#define DBG_EQUI_WRITE
#ifdef DBG_EQUI_WRITE
#define DBG_EQUI_WRITE_FN(func) func
#else
#define DBG_EQUI_WRITE_FN(func)
#endif

#define EQUI_TIMER
#ifdef EQUI_TIMER
#define EQUI_TIMER_DEFINE std::chrono::high_resolution_clock::time_point eq_timer_start, eq_timer_stop;
#define EQUI_TIMER_START eq_timer_start = std::chrono::high_resolution_clock::now();
#define EQUI_TIMER_STOP(operation)  eq_timer_stop = std::chrono::high_resolution_clock::now(); \
    std::cout << operation << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(eq_timer_stop - eq_timer_start).count() << " ms" << std::endl;
#else
#define EQUI_TIMER_DEFINE()
#define EQUI_TIMER_START()
#define EQUI_TIMER_STOP(operation)
#endif

template<typename EquihashType>
class EhDevice
{
public:
    EhDevice();
    ~EhDevice() {}

    std::unique_ptr<blake2b_state, CudaDeleter> initialState;
    std::unique_ptr<uint32_t, CudaDeleter> hashes;               // NBucketCount * NBucketSizeExtra * HashWords
    std::unique_ptr<uint32_t, CudaDeleter> xoredHashes;          // NBucketCount * NBucketSizeExtra * HashWords
    std::unique_ptr<uint32_t, CudaDeleter> bucketHashIndices;    // NBucketCount * NBucketSizeExtra * (WK + 1)
    std::unique_ptr<uint32_t, CudaDeleter> bucketHashCounters;   // NBucketCount * (WK + 1)

    std::unique_ptr<uint32_t, CudaDeleter> discardedCounter;     // 1
    std::unique_ptr<uint32_t, CudaDeleter> collisionPairs;       // NBucketCount * MaxCollisionsPerBucket
    // Accumulated collision pair offsets for each bucket
    v_uint32 vCollisionPairsOffsets;                             // NBucketCount
    std::unique_ptr<uint32_t, CudaDeleter> collisionOffsets;     // NBucketCount * (WK + 1)
    std::unique_ptr<uint32_t, CudaDeleter> collisionCounters;    // NBucketCount
    v_uint32 vCollisionCounters;                                 // NBucketCount

    std::unique_ptr<typename EquihashType::solution_type, CudaDeleter> solutions; // MAXSOLUTIONS
    std::unique_ptr<uint32_t, CudaDeleter> solutionCount;        // 1

    uint32_t round = 0;

    bool allocate_memory();
    uint32_t solver();

    void copySolutionsToHost(std::vector<typename EquihashType::solution_type>& vHostSolutions);
    void debugWriteSolutions(const std::vector<typename EquihashType::solution_type>& vHostSolutions);
    void debugTraceSolution(const uint32_t bucketIdx);

    static inline constexpr uint32_t ThreadsPerBlock = 256;
    static inline constexpr uint32_t MaxCollisionsPerBucket = 10'000;
    static inline constexpr uint32_t MaxSolutions = 100;

private:
    void generateInitialHashes();
    void processCollisions();
    uint32_t findSolutions();

    void debugPrintHashes(const bool bInitialHashes);
    void debugPrintCollisionPairs();
    void debugPrintBucketCounters(const uint32_t bucketIdx, const uint32_t *collisionCountersPtr);

    void debugWriteHashes(const bool bInitialHashes);
    void debugWriteCollisionPairs();
    void debugWriteBucketIndices();
    std::ofstream m_dbgFile;
};
