#pragma once

#include <bitset>
#include <iostream>

#define PROBE_VERSION 1

#define ERROR_CHECK 0
#define DEBUG_PRINT 0
#define FILTER_VERSION 0

#ifndef BENCHMARK_TIME
#define BENCHMARK_TIME 0
#endif

#ifndef SHARED_MEM_HASH_TABLE
#define SHARED_MEM_HASH_TABLE 1
#endif

#define BENCHMARK_PART 0
#define PROBE_MODE 0
#define EXTRACT_MODE 0

// memory free tolerance in bytes
#define MEMORY_TOLERANCE 500000000;

// 32 or 64 bit unsigned int
#define HASH_BITS 64

// 8, 32, 64, 128 bit chunks
#define HASH_CHUNK_BITS 64
