/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * This file is based on code from Flashinfer,
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/math.cuh
 * Copyright (c) 2023 by FlashInfer team.
 * Small modifications made by SageAttention team, 2024 (e.g., renamed
 * namespace).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// will have to translate assembly manually

#pragma once
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#ifndef USHORT_TYPE
        #define USHORT_TYPE
typedef unsigned short ushort;
#endif

namespace math {

// log2(e)
constexpr float log2e      = 1.44269504088896340736f;
constexpr float log2e_recp = 1.0f / log2e;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2 *)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t *)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) { return exp2f(x); }

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) { return log2f(x); }

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) { return h2exp2(x); }

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) { return hexp2(x); }

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) { return 1.0f / x; }

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly
 * shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^
 * delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) { return __shfl_xor(x, lane_mask); }

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a
 * butterfly shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^
 * lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) { return __shfl_xor(x, lane_mask); }

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes
 * 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
        // ROCm automatically maps this to the optimal v_rsq_f32 hardware instruction
        return rsqrtf(x);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
        // ROCm automatically maps this to the v_tanh_f32 hardware instruction
        return tanhf(x);
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
        // Unpack, run fast float math, and repack. LLVM optimizes this heavily.
        float2 x_f2 = __half22float2(x);
        return __float22half2_rn(make_float2(tanhf(x_f2.x), tanhf(x_f2.y)));
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) { return __float2half(tanhf(__half2float(x))); }

} // namespace math
