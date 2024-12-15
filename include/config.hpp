#pragma once

#include <cstdint>
#include <experimental/simd>

// specific number types used by neurons
typedef long double number;

#define MAX_SIMD_VECTOR_SIZE std::experimental::simd_abi::max_fixed_size<number>

typedef std::experimental::fixed_size_simd<number , MAX_SIMD_VECTOR_SIZE> SIMD;

typedef std::experimental::fixed_size_simd_mask<number , MAX_SIMD_VECTOR_SIZE> SIMD_MASK;

#define MAITING_THRESHOLD 0.5f

#define AMOUNT_THAT_PASS 0.3f

#define USESES_TO_MAITING 4

#define MAX_THREAD_POOL 8

#define SWARMING_SPEED_DEFAULT 10.f

#define INITIAL_STD 0.1f

#define MIN_STD 0.00001f

#define MAX_STD 0.5f

#define MUTATION_PROBABILITY 0.25f

#define REWARD_TO_SWITCH_PROBABILITY 0.0005f

#define REWARD_DIFFERENCE_GAIN 0 // 10

// Reward PID:

#define PID_DELTA 0.1f

#define P_REWARD 1.f

#define I_REWARD 0.01f

#define D_REWARD 0.001f


#define USED_THREADS 4