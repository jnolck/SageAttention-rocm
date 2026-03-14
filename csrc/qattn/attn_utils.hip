/*
 * Copyright (c) 2024 by SageAttention team.
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

#pragma once
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include "../math.hip"
#include "../utils.hip"
#include <hip/hip_fp16.h>
// #include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "../cp_async.hip"
#include "../mma.hip"
#include "../numeric_conversion.hip"
#include "../permuted_smem.hip"

#define WARP_SIZE 32

#define S_FP8_OFFSET 8.807f
#define S_FP8_OFFSET_EXP 6680.8477f
#define S_FP8_OFFSET_EXP_INV 0.0022326917f

#define div_ceil(M, N) (((M) + (N) - 1) / (N))

enum class MaskMode {
	kNone	= 0,
	kCausal = 1,
};

enum class DataType {
	kHalf,
	kInt8,
	kInt4,
	kE4M3,
	kE5M2,
};

enum class QuantGranularity {
	kPerTensor = 0,
	kPerBlock  = 1,
	kPerWarp   = 2,
	kPerThread = 3,
};

enum class ComputeUnit {
	kTensorCore,
	kCudaCore,
};

__device__ __forceinline__ uint32_t get_warp_id()
{
	return threadIdx.y;
}

__device__ __forceinline__ uint32_t get_lane_id()
{
	return threadIdx.x;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_q()
{
	return get_warp_id() / num_warps_k;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_k()
{
	return get_warp_id() % num_warps_k;
}

template <uint32_t    global_to_shared_line_lanes,
	  uint32_t    global_to_shared_copy_lines_per_warp_per_iter,
	  uint32_t    smem_iters_row,
	  uint32_t    smem_iters_col,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  uint32_t    CTA,
	  typename T>
__device__ __forceinline__ void load_global_to_share(T **lane_ptr, uint32_t &smem_offset, const uint32_t &gmem_stride, const smem_t<swizzle_mode, stride> &smem)
{
	static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
	static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

	constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

	//clang-format off
#pragma unroll
	for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
		for (uint32_t j = 0; j < smem_iters_row; j++) {
			smem.load_128b_async(smem_offset, *lane_ptr);
			*lane_ptr += (global_to_shared_line_lanes * pack_size);
			smem_offset = smem.template advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
		}
		//clang-format on
		smem_offset = smem.template advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
			smem_offset - (smem_iters_row * global_to_shared_line_lanes));
		*lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
	}
	smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
	*lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

// with predicate
template <uint32_t    global_to_shared_line_lanes,
	  uint32_t    global_to_shared_copy_lines_per_warp_per_iter,
	  uint32_t    smem_iters_row,
	  uint32_t    smem_iters_col,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  uint32_t    CTA,
	  typename T>
__device__ __forceinline__ void load_global_to_share(T				       **lane_ptr,
						     uint32_t				&smem_offset,
						     const uint32_t			&gmem_stride,
						     const smem_t<swizzle_mode, stride> &smem,
						     uint32_t				 base_idx,
						     uint32_t				 max_len)
{
	static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
	static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

	constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
	for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
		for (uint32_t j = 0; j < smem_iters_row; j++) {
			smem.template load_128b_async<cp_async::SharedMemFillMode::kNoFill>(smem_offset, *lane_ptr, base_idx < max_len);
			*lane_ptr += (global_to_shared_line_lanes * pack_size);
			smem_offset = smem.template advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
		}

		smem_offset = smem.template advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
			smem_offset - (smem_iters_row * global_to_shared_line_lanes));
		*lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size));
		base_idx += global_to_shared_copy_lines_per_warp_per_iter;
	}
	smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
	*lane_ptr += (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <uint32_t    global_to_shared_line_lanes,
	  uint32_t    global_to_shared_copy_lines_per_warp_per_iter,
	  uint32_t    smem_iters_row,
	  uint32_t    smem_iters_col,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  uint32_t    CTA>
__device__ __forceinline__ void
load_fp8_V_global_to_share(int8_t **lane_ptr, uint32_t &smem_offset, const uint32_t &gmem_stride, const smem_t<swizzle_mode, stride> &smem)
{
	static_assert(global_to_shared_copy_lines_per_warp_per_iter * global_to_shared_line_lanes == WARP_SIZE);
	constexpr uint32_t pack_size_fp8 = 16;

#pragma unroll
	for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
		for (uint32_t j = 0; j < smem_iters_row; j++) {
			smem.load_128b_async(smem_offset, *lane_ptr);
			*lane_ptr += (global_to_shared_line_lanes * pack_size_fp8);
			smem_offset = smem.template advance_offset_by_column<global_to_shared_line_lanes>(smem_offset);
		}

		smem_offset = smem.template advance_offset_by_row<global_to_shared_copy_lines_per_warp_per_iter>(
			smem_offset - (smem_iters_row * global_to_shared_line_lanes));
		*lane_ptr += ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) - (smem_iters_row * global_to_shared_line_lanes * pack_size_fp8));
	}
	smem_offset -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
	// for QK: *lane_ptr += (CTA - smem_iters_col *
	// global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
	*lane_ptr += CTA; // ! prevent underflow
	*lane_ptr -= (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
}

template <uint32_t    num_warps_q,
	  uint32_t    num_warps_k,
	  uint32_t    num_tiles_q,
	  uint32_t    num_tiles_k,
	  uint32_t    num_tiles_qk_inner,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  DataType    DTypeQK>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_Q,
					       const smem_t<swizzle_mode, stride> &smem_K,
					       int32_t				   RS[][num_tiles_k][8],
					       uint32_t				  &offset_Q,
					       uint32_t				  &offset_K)
{
	uint32_t RQ[num_tiles_q][4];
	uint32_t RK[4];

	// ==========================================================
	// ITERATION 0: mma mode is kInit (Zero out the accumulators)
	// ==========================================================
#pragma unroll
	for (uint32_t iter = 0; iter < 1; iter++) {
		// 1. Load RQ
#pragma unroll
		for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
			auto	&fragA	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, int8_t, rocwmma::row_major> &>(RQ[fq]);
			uint32_t warp_base_Q = __shfl(offset_Q, 0);
			rocwmma::load_matrix_sync(fragA, (const int8_t *)smem_Q.base + warp_base_Q * 16, stride * 16);
			offset_Q = smem_Q.template advance_offset_by_row<16>(offset_Q);
		}
		offset_Q = smem_Q.template advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

		// 2. Load RK and Execute Math
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			auto	&fragB	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, int8_t, rocwmma::col_major> &>(RK);
			uint32_t warp_base_K = __shfl(offset_K, 0);
			rocwmma::load_matrix_sync(fragB, (const int8_t *)smem_K.base + warp_base_K * 16, stride * 16);
			offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

#pragma unroll
			for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
				auto &fragA = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, int8_t, rocwmma::row_major> &>(RQ[fq]);
				auto &fragC = reinterpret_cast<rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, int32_t> &>(RS[fq][fk]);
				rocwmma::fill_fragment(fragC,
						       0); // kInit: Zero out
				rocwmma::mma_sync(fragC, fragA, fragB, fragC);
			}
		}
		offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
	}

	// ==========================================================
	// FOLLOWING ITERATIONS: mma mode is kInplaceUpdate
	// ==========================================================
#pragma unroll
	for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
		// 1. Load RQ
#pragma unroll
		for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
			auto	&fragA	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, int8_t, rocwmma::row_major> &>(RQ[fq]);
			uint32_t warp_base_Q = __shfl(offset_Q, 0);
			rocwmma::load_matrix_sync(fragA, (const int8_t *)smem_Q.base + warp_base_Q * 16, stride * 16);
			offset_Q = smem_Q.template advance_offset_by_row<16>(offset_Q);
		}
		offset_Q = smem_Q.template advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);

		// 2. Load RK and Execute Math
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			auto	&fragB	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, int8_t, rocwmma::col_major> &>(RK);
			uint32_t warp_base_K = __shfl(offset_K, 0);
			rocwmma::load_matrix_sync(fragB, (const int8_t *)smem_K.base + warp_base_K * 16, stride * 16);
			offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

#pragma unroll
			for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
				auto &fragA = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, int8_t, rocwmma::row_major> &>(RQ[fq]);
				auto &fragC = reinterpret_cast<rocwmma::fragment<rocwmma::accumulator, 16, 16, 32, int32_t> &>(RS[fq][fk]);
				// kInplace: Accumulate on top of existing values
				rocwmma::mma_sync(fragC, fragA, fragB, fragC);
			}
		}
		offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
	}

	offset_Q -= (2 * num_tiles_qk_inner);
	offset_K -= (2 * num_tiles_qk_inner);
}

// for case when num_tiles_qk_inner = 1
template <uint32_t    num_warps_q,
	  uint32_t    num_warps_k,
	  uint32_t    num_tiles_q,
	  uint32_t    num_tiles_k,
	  uint32_t    num_tiles_qk_inner,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  DataType    DTypeQK>
__device__ __forceinline__ void compute_int_qk(const smem_t<swizzle_mode, stride> &smem_K, int32_t RS[][num_tiles_k][8], uint32_t RQ[][4], uint32_t offset_K)
{
	static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4);
	static_assert(num_tiles_qk_inner == 1);

	uint32_t RK[4];

	// mma mode is kInit
#pragma unroll
	for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
		// load RK
#ifdef __HIP_PLATFORM_AMD__
		auto	&fragB	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_b, 16, 16, 32, int8_t, rocwmma::col_major> &>(RK);
		uint32_t warp_base_K = __shfl(offset_K, 0);
		rocwmma::load_matrix_sync(fragB, (const int8_t *)smem_K.base + warp_base_K * 16, stride * 16);
#else
		smem_K.ldmatrix_m8n8x4(offset_K, RK);
#endif
		offset_K = smem_K.template advance_offset_by_row<16>(offset_K);

		// mma
#pragma unroll
		for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
			if constexpr (DTypeQK == DataType::kInt8) {
#ifndef __HIP_PLATFORM_AMD__
				mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
#endif
			} else if constexpr (DTypeQK == DataType::kInt4) {
#ifndef __HIP_PLATFORM_AMD__
				mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(RS[fq][fk], RQ[fq], RK);
#endif
			}
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
__device__ __forceinline__ void apply_causal_mask(const uint32_t &Q_idx_lane_base, const uint32_t &K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8])
{
	uint32_t lane_id     = get_lane_id();
	uint32_t q_base	     = Q_idx_lane_base - (lane_id / 4);
	uint32_t k_base	     = K_idx_lane_base - 2 * (lane_id % 4);

	uint32_t col_in_frag = lane_id % 16;
	uint32_t row_offset  = (lane_id < 16) ? 0 : 1;

#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			uint32_t kv_idx = k_base + fk * 16 + col_in_frag;

			if constexpr (std::is_same<DTypeQKAccum, float>::value) {
				// NO PRAGMA UNROLL HERE
				for (uint32_t k = 0; k < 8; k++) {
					uint32_t row_in_frag = (k * 2) + row_offset;
					uint32_t q_idx	     = q_base + fq * 16 + row_in_frag;

					RS[fq][fk][k]	     = ((kv_idx > q_idx) ? -5000000.0f : RS[fq][fk][k]);
				}
			}
		}
	}
}
template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename DTypeQKAccum>
__device__ __forceinline__ void apply_out_of_bound_mask(const uint32_t &K_idx_lane_base, DTypeQKAccum RS[][num_tiles_k][8], const uint32_t &kv_len)
{
	uint32_t lane_id     = get_lane_id();
	uint32_t k_base	     = K_idx_lane_base - 2 * (lane_id % 4);
	uint32_t col_in_frag = lane_id % 16;

#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			uint32_t   kv_idx	   = k_base + fk * 16 + col_in_frag;
			const bool out_of_boundary = (kv_idx >= kv_len);

			if constexpr (std::is_same<DTypeQKAccum, float>::value) {
				// NO PRAGMA UNROLL HERE
				for (uint32_t k = 0; k < 8; k++) {
					RS[fq][fk][k] = (out_of_boundary ? -5000000.0f : RS[fq][fk][k]);
				}
			}
		}
	}
}

// for DTypeQKAccum float
template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, bool use_half_o_scale, bool exp_offset, bool fuse_scale = false, typename DTypeSVAccum>
__device__ __forceinline__ void update_mdo(float RS[][num_tiles_k][8], DTypeSVAccum RO[][num_tiles_v][8], float m[][8], float d[][8], const float &sm_scale)
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		// Execute Softmax independently for each of the 8 rows in the thread
		for (int k = 0; k < 8; k++) {
			float m_prev = m[fq][k];
			float m_temp = -5000000.0f;
#pragma unroll
			for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
				m_temp = max(m_temp, RS[fq][fk][k]);
			}

			// RDNA3 Row Max: Reduce across the half-warp (Threads 0-15 own Cols 0-15 of the row)
			m_temp = max(m_temp, __shfl_xor(m_temp, 8));
			m_temp = max(m_temp, __shfl_xor(m_temp, 4));
			m_temp = max(m_temp, __shfl_xor(m_temp, 2));
			m_temp = max(m_temp, __shfl_xor(m_temp, 1));

			m_temp *= sm_scale;
			m[fq][k]      = max(m[fq][k], m_temp);

			float o_scale = math::ptx_exp2(m_prev - m[fq][k]);
			d[fq][k] *= o_scale;

#pragma unroll
			for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
				RO[fq][fv][k] *= o_scale;
			}

			float negative_m = -m[fq][k];
#pragma unroll
			for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
				RS[fq][fk][k] = math::ptx_exp2(fmaf(RS[fq][fk][k], sm_scale, negative_m));
			}
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void RS_32_to_16(T RS[][num_tiles_k][8], uint32_t RS_16[][num_tiles_k][4])
{
	static_assert(sizeof(T) == 4);
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			((half2 *)RS_16[fq][fk])[0] = __float22half2_rn(((float2 *)RS[fq][fk])[0]);
			((half2 *)RS_16[fq][fk])[1] = __float22half2_rn(((float2 *)RS[fq][fk])[1]);
			((half2 *)RS_16[fq][fk])[2] = __float22half2_rn(((float2 *)RS[fq][fk])[2]);
			((half2 *)RS_16[fq][fk])[3] = __float22half2_rn(((float2 *)RS[fq][fk])[3]);
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_32_to_8(float RS[][num_tiles_k][8], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
			floatx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 4);
			floatx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2, RS[fq][fk * 2 + 0] + 6);
			floatx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 4);
			floatx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2, RS[fq][fk * 2 + 1] + 6);
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_16_to_8(uint32_t RS[][num_tiles_k][4], uint32_t RS_8[][num_tiles_k / 2][4])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
			halfx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2);
			halfx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3);
			halfx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2);
			halfx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3);
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_8_to_16(uint32_t RS_8[][num_tiles_k / 2][4], uint32_t RS[][num_tiles_k][4])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
			e4m3x4_to_halfx4(RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2, RS_8[fq][fk]);
			e4m3x4_to_halfx4(RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3, RS_8[fq][fk] + 1);
			e4m3x4_to_halfx4(RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2, RS_8[fq][fk] + 2);
			e4m3x4_to_halfx4(RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3, RS_8[fq][fk] + 3);
		}
	}
}

// Float overload
template <uint32_t num_tiles_q, uint32_t num_tiles_k, ComputeUnit compute_unit = ComputeUnit::kCudaCore, typename T>
__device__ __forceinline__ void accumulate_d(T RS[][num_tiles_k][8], float d[][8])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (int k = 0; k < 8; k++) {
			float sum = 0.0f;
#pragma unroll
			for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
				sum += (float)RS[fq][fk][k];
			}
			// RDNA3 Row Sum: Reduce across the half-warp
			sum += __shfl_xor(sum, 8);
			sum += __shfl_xor(sum, 4);
			sum += __shfl_xor(sum, 2);
			sum += __shfl_xor(sum, 1);

			d[fq][k] += sum;
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, ComputeUnit compute_unit = ComputeUnit::kCudaCore, typename T>
__device__ __forceinline__ void accumulate_d(T RS[][num_tiles_k][4], float d[][8])
{
}

template <uint32_t num_tiles_q, uint32_t num_tiles_v, ComputeUnit compute_unit = ComputeUnit::kCudaCore, typename DTypeSVAccum>
__device__ __forceinline__ void normalize_d(DTypeSVAccum RO[][num_tiles_v][8], float m[][8], float d[][8])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (int k = 0; k < 8; k++) {
			// Calculate the inverse of the denominator once per row
			float d_inv = 1.0f / d[fq][k];

#pragma unroll
			for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
				RO[fq][fv][k] *= d_inv;
			}
		}
	}
}

// RDNA 3 Overload for raw 32-bit Float Arrays (Size 8)
template <uint32_t num_tiles_q, uint32_t num_tiles_k, ComputeUnit compute_unit = ComputeUnit::kCudaCore, typename T>
__device__ __forceinline__ void accumulate_d(T RS[][num_tiles_k][8], float d[][2])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
			float sum = 0.0f;
#pragma unroll
			for (int k = 0; k < 8; k++) {
				sum += (float)RS[fq][fk][k];
			}
			// RDNA3 Row Sum: Thread i and Thread i+16 share the same Row!
			d[fq][0] += sum + __shfl_xor(sum, 16);
		}
	}
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void accumulate_d_f8(uint32_t RS[][num_tiles_k / 2][4], float d[][2])
{
#pragma unroll
	for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
		for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
#ifndef __HIP_PLATFORM_AMD__
			mma::rowsum_f8f8f32(d[fq], RS[fq][fk]);
#endif
		}
	}
}

template <uint32_t    num_warps_q,
	  uint32_t    num_warps_k,
	  uint32_t    num_tiles_q,
	  uint32_t    num_tiles_k,
	  uint32_t    num_tiles_v,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp16_sv(const smem_t<swizzle_mode, stride> &smem_V, uint32_t RS_f16[][num_tiles_k][4], DTypeSVAccum RO[][num_tiles_v][8], float d[][2])
{
	uint32_t smem_V_row_base = get_warp_idx_k<num_warps_q, num_warps_k>() * (num_tiles_k * 16) + get_lane_id() % 16;
	uint32_t smem_V_col_base = get_lane_id() / 16;
#pragma unroll
	for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
		for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
			// load RV
			uint32_t RV[4];
			uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fk * 16, smem_V_col_base + fv * 2);
#ifdef __HIP_PLATFORM_AMD__
			auto	&fragB	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> &>(RV);
			uint32_t warp_base_V = __shfl(offset_V, 0);
			rocwmma::load_matrix_sync(fragB, (const __half *)smem_V.base + warp_base_V * 8, stride * 8);
#else
			smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#endif
#pragma unroll
			for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
				if constexpr (std::is_same<DTypeSVAccum, float>::value) {
#ifndef __HIP_PLATFORM_AMD__
					mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], RS_f16[fq][fk], RV);
#endif
				} else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
#ifndef __HIP_PLATFORM_AMD__
					mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t *)RO[fq][fv], RS_f16[fq][fk], RV);
#endif
				}
			}
		}
	}
}

template <uint32_t    num_warps_q,
	  uint32_t    num_warps_k,
	  uint32_t    num_tiles_q,
	  uint32_t    num_tiles_k,
	  uint32_t    num_tiles_v,
	  SwizzleMode swizzle_mode,
	  uint32_t    stride,
	  uint32_t    RS_width = 4,
	  typename T,
	  typename DTypeSVAccum>
__device__ __forceinline__ void compute_fp16_sv_permuted(const smem_t<swizzle_mode, stride> &smem_V,
							 T				     RS_f16[][num_tiles_k][RS_width],
							 DTypeSVAccum			     RO[][num_tiles_v][8],
							 float				     d[][8],
							 uint32_t			    &offset_V)
{
	uint32_t	lane_id = get_lane_id();

	// 1. Declare the static scratchpad at the TOP of the function
	__shared__ half a_scratch[32][256];
	uint32_t	global_warp_id = (threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) / 32;

#pragma unroll
	for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
		for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
			uint32_t RV[4];
			auto	&fragB	     = reinterpret_cast<rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> &>(RV);
			uint32_t warp_base_V = __shfl(offset_V, 0);
			rocwmma::load_matrix_sync(fragB, (const __half *)smem_V.base + warp_base_V * 8, stride * 8);

#pragma unroll
			for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
				// 2. Cast RS_f16 to accumulator fragment (Hardware knows it's vertical Even/Odd stripes)
				auto &fragC_rs = reinterpret_cast<rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, __half> &>(RS_f16[fq][fk]);

				// 3. Store to static scratchpad (Hardware descrambles it to a pure Row-Major grid)
				rocwmma::store_matrix_sync(a_scratch[global_warp_id], fragC_rs, 16, rocwmma::mem_row_major);

				// 4. Load back as Matrix A (Hardware reads it into the required horizontal stripes)
				rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> fragA;
				rocwmma::load_matrix_sync(fragA, a_scratch[global_warp_id], 16);

				// BUG 1 FIX: Match the accumulator type to DTypeSVAccum!
				if constexpr (std::is_same<DTypeSVAccum, float>::value) {
					auto &fragC = reinterpret_cast<rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> &>(RO[fq][fv]);
					rocwmma::mma_sync(fragC, fragA, fragB, fragC);
				} else {
					auto &fragC = reinterpret_cast<rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, __half> &>(RO[fq][fv]);
					rocwmma::mma_sync(fragC, fragA, fragB, fragC);
				}
			}
			offset_V = smem_V.template advance_offset_by_column<2>(offset_V, fv);
		}
		offset_V = smem_V.template advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
	}
	offset_V -= (16 * num_tiles_k * stride);
}

// template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
//           uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode
//           swizzle_mode, uint32_t stride, typename DTypeSVAccum>
// __device__ __forceinline__ void
// compute_fp8_sv(const smem_t<swizzle_mode, stride> &smem_V,
//                uint32_t RS_f8[][num_tiles_k / 2][4],
//                DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
//   uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
//   uint32_t smem_V_col_base =
//       get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16)
//       + (get_lane_id() / 8) % 2;
// #pragma unroll
//   for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
//     uint32_t offset_V =
//         smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk *
//         2);
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // load RV
//       uint32_t RV[4];
//       uint32_t offset_V = (smem_V).get_permuted_offset(
//           smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
//       smem_V.ldmatrix_m8n8x4(offset_V, RV);
// #pragma unroll
//       for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
//         if constexpr (std::is_same<DTypeSVAccum, float>::value) {
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f32(RO[fq][fv], RS_f8[fq][fk],
//                                                   RV);
// #endif
//         } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
//           // ! Not Implemented
//         }
//       }
//       offset_V = smem_V.template advance_offset_by_row<16>(offset_V);
//     }
//   }
// }

// template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
//           uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode
//           swizzle_mode, uint32_t stride, typename DTypeSVAccum>
// __device__ __forceinline__ void
// compute_fp8_sv_inst_buf(const smem_t<swizzle_mode, stride> &smem_V,
//                         uint32_t RS_f8[][num_tiles_k / 2][4],
//                         DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
//   uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
//   uint32_t smem_V_col_base =
//       get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16)
//       + (get_lane_id() / 8) % 2;
//
//   float RO_inst_buf[num_tiles_q][num_tiles_v][8];
//
// #pragma unroll
//   for (uint32_t fk = 0; fk < 1; fk++) {
//     uint32_t offset_V =
//         smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk *
//         2);
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // load RV
//       uint32_t RV[4];
//       uint32_t offset_V = (smem_V).get_permuted_offset(
//           smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
//       smem_V.ldmatrix_m8n8x4(offset_V, RV);
// #pragma unroll
//       for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
//         if constexpr (std::is_same<DTypeSVAccum, float>::value) {
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(
//               RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
// #endif
//         } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
//           // ! Not Implemented
//         }
//       }
//       offset_V = smem_V.template advance_offset_by_row<16>(offset_V);
//     }
//   }
//
// #pragma unroll
//   for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
//     uint32_t offset_V =
//         smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk *
//         2);
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // load RV
//       uint32_t RV[4];
//       uint32_t offset_V = (smem_V).get_permuted_offset(
//           smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
//       smem_V.ldmatrix_m8n8x4(offset_V, RV);
// #pragma unroll
//       for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
//         if constexpr (std::is_same<DTypeSVAccum, float>::value) {
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(
//               RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
// #endif
//         } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
//           // ! Not Implemented
//         }
//       }
//       offset_V = smem_V.template advance_offset_by_row<16>(offset_V);
//     }
//   }
//
// #pragma unroll
//   for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
//       RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
//       RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
//       RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
//       RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
//       RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
//       RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
//       RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
//     }
//   }
// }

// template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
//           uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode
//           swizzle_mode, uint32_t stride, typename DTypeSVAccum>
// __device__ __forceinline__ void
// compute_fp8_sv_inst_buf_fp16_accu(const smem_t<swizzle_mode, stride> &smem_V,
//                                   uint32_t RS_f8[][num_tiles_k / 2][4],
//                                   DTypeSVAccum RO[][num_tiles_v][8],
//                                   float d[][2]) {
//   uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
//   uint32_t smem_V_col_base =
//       get_warp_idx_k<num_warps_q, num_warps_k>() * ((16 * num_tiles_k) / 16)
//       + (get_lane_id() / 8) % 2;
//
//   uint32_t RO_int32[num_tiles_q][num_tiles_v][4];
//
// #pragma unroll
//   for (uint32_t fk = 0; fk < 1; fk++) {
//     uint32_t offset_V =
//         smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk *
//         2);
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // load RV
//       uint32_t RV[4];
//       uint32_t offset_V = (smem_V).get_permuted_offset(
//           smem_V_row_base + fv * 16, smem_V_col_base + fk * 2);
//       smem_V.ldmatrix_m8n8x4(offset_V, RV);
// #pragma unroll
//       for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
//         if constexpr (std::is_same<DTypeSVAccum, float>::value) {
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(
//               RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
// #endif
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInit>(
//               RO_int32[fq][fv], RS_f8[fq][fk], RV);
// #endif
//         } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
//           // ! Not Implemented
//         }
//       }
//       offset_V = smem_V.template advance_offset_by_row<16>(offset_V);
//     }
//   }
//
// #pragma unroll
//   for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
//     uint32_t offset_V =
//         smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk *
//         2);
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // load RV
//       uint32_t RV[4];
//       uint32_t offset_V = 0; //(smem_V).get_permuted_offset(smem_V_row_base +
//       fv
//                              //* 16, smem_V_col_base + fk * 2);
//       smem_V.ldmatrix_m8n8x4(offset_V, RV);
// #pragma unroll
//       for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
//         if constexpr (std::is_same<DTypeSVAccum, float>::value) {
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(
//               RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
// #endif
// #ifndef __HIP_PLATFORM_AMD__
//           mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInplaceUpdate>(
//               RO_int32[fq][fv], RS_f8[fq][fk], RV);
// #endif
//         } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
//           // ! Not Implemented
//         }
//       }
//       offset_V = smem_V.template advance_offset_by_row<16>(offset_V);
//     }
//   }
//   float RO_tmp_float[2];
// #pragma unroll
//   for (int i = 0; i < num_tiles_q; i++) {
// #pragma unroll
//     for (int j = 0; j < num_tiles_v; j++) {
// #pragma unroll
//       for (int k = 0; k < 4; k++) {
//         unpack_half2_from_uint32_to_float(RO_tmp_float, RO_int32[i][j][k]);
//         RO[i][j][k * 2 + 0] += RO_tmp_float[0];
//         RO[i][j][k * 2 + 1] += RO_tmp_float[1];
//       }
//     }
//   }
//
// #pragma unroll
//   for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
// #pragma unroll
//     for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
//       // RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
//       // RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
//       // RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
//       // RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
//       // RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
//       // RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
//       // RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
//       // RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
//     }
//   }
// }
