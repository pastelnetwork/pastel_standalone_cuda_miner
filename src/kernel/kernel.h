// Copyright (c) 2024 The Pastel developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://www.opensource.org/licenses/mit-license.php.
#pragma once
#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <atomic>

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

//#define EQUI_TIMER
#ifdef EQUI_TIMER
#define EQUI_TIMER_DEFINE std::chrono::high_resolution_clock::time_point eq_timer_start, eq_timer_stop;
#define EQUI_TIMER_START eq_timer_start = std::chrono::high_resolution_clock::now();
#define EQUI_TIMER_STOP(operation)  eq_timer_stop = std::chrono::high_resolution_clock::now(); \
    debug("{}: {} ms", operation, std::chrono::duration_cast<std::chrono::milliseconds>(eq_timer_stop - eq_timer_start).count());
#define EQUI_TIMER_DEFINE_EX(name) std::chrono::high_resolution_clock::time_point eq_timer_start_##name, eq_timer_stop_##name;
#define EQUI_TIMER_START_EX(name) eq_timer_start_##name = std::chrono::high_resolution_clock::now();
#define EQUI_TIMER_STOP_EX(name, operation)  eq_timer_stop_##name = std::chrono::high_resolution_clock::now(); \
    debug("{}: {} ms", operation, std::chrono::duration_cast<std::chrono::milliseconds>(eq_timer_stop_##name - eq_timer_start_##name).count());
#else
#define EQUI_TIMER_DEFINE
#define EQUI_TIMER_START
#define EQUI_TIMER_STOP(operation)
#define EQUI_TIMER_DEFINE_EX(name)
#define EQUI_TIMER_START_EX(name)
#define EQUI_TIMER_STOP_EX(name, operation)
#endif

struct MaxGridSize
{
    int x, y, z;
};

template<typename EquihashType>
class EhDevice
{
public:
    EhDevice(bool bNewSolver);
    ~EhDevice() {}

    std::unique_ptr<blake2b_state, CudaDeleter> d_initialState;
    std::unique_ptr<uint32_t, CudaDeleter> d_hashes;               // NBucketCount * NBucketSizeExtra * HashWords
    std::unique_ptr<uint32_t, CudaDeleter> d_xoredHashes;          // NBucketCount * NBucketSizeExtra * HashWords
    std::unique_ptr<uint32_t, CudaDeleter> d_bucketHashIndices;    // NBucketCount * NBucketSizeExtra * (WK + 1)
    std::unique_ptr<uint32_t, CudaDeleter> d_bucketHashCounters;   // NBucketCount * (WK + 1)

    std::unique_ptr<uint32_t, CudaDeleter> d_discardedCounter;     // 1
    std::unique_ptr<uint32_t, CudaDeleter> d_collisionPairs;       // NBucketCount * MaxCollisionsPerBucket
    std::unique_ptr<uint32_t, CudaDeleter> d_collisionOffsets;     // NBucketCount * (WK + 1)
    std::unique_ptr<uint32_t, CudaDeleter> d_collisionCounters;    // NBucketCount

    std::unique_ptr<typename EquihashType::solution_type, CudaDeleter> d_solutions; // MaxSolutions
    std::unique_ptr<uint32_t, CudaDeleter> d_solutionCount;

    // Accumulated collision pair offsets for each bucket
    v_uint32 vCollisionPairsOffsets;                               // NBucketCount
    v_uint32 vCollisionCounters;                                   // NBucketCount

    uint32_t round = 0;
	std::atomic_bool bBreakSolver;

    uint32_t solve();

    void copySolutionsToHost(std::vector<typename EquihashType::solution_type>& vHostSolutions);
    void debugWriteSolutions(const std::vector<typename EquihashType::solution_type>& vHostSolutions);
    
    static inline constexpr uint32_t MaxCollisionsPerBucket = 10'000;
    static inline constexpr uint32_t MaxSolutions = 20;

private:
    void generateInitialHashes();
    void processCollisions();
    uint32_t findSolutions();
    bool allocate_memory();
    void clear();

    void debugPrintHashes(const bool bInitialHashes);
    void debugPrintCollisionPairs();
    void debugPrintBucketCounters(const uint32_t bucketIdx, const uint32_t *collisionCountersPtr);

    void debugWriteHashes(const bool bInitialHashes);
    void debugWriteCollisionPairs();
    void debugWriteBucketIndices();
    std::ofstream m_dbgFile;

	bool bUseNewSolver = false;
    int nCudaDeviceCount;
	uint32_t nCudaMaxThreads;
	MaxGridSize CudaMaxGridSize;
};
