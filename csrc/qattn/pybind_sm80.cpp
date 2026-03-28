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

#include "attn_cuda_sm80.hip"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// 1. THE TROJAN HORSE: Keeps Python's import system happy
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}
// TODO: Refactor host signatures natively for PyTorch 2.10
// 2. THE 64-BIT ADAPTERS: PyTorch 2.10 demands int64_t and double at the boundary.
// We accept them here, cast them to 32-bit, and hand them to the fast RDNA3 kernels.
torch::Tensor wrap_f32_attn(torch::Tensor query,
			    torch::Tensor key,
			    torch::Tensor value,
			    torch::Tensor output,
			    torch::Tensor query_scale,
			    torch::Tensor key_scale,
			    int64_t	  tensor_layout,
			    int64_t	  is_causal,
			    int64_t	  qk_quant_gran,
			    double	  sm_scale,
			    int64_t	  return_lse)
{
	return qk_int8_sv_f16_accum_f32_attn(query, key, value, output, query_scale, key_scale, static_cast<int>(tensor_layout), static_cast<int>(is_causal),
					     static_cast<int>(qk_quant_gran), static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

torch::Tensor wrap_f16_attn(torch::Tensor query,
			    torch::Tensor key,
			    torch::Tensor value,
			    torch::Tensor output,
			    torch::Tensor query_scale,
			    torch::Tensor key_scale,
			    int64_t	  tensor_layout,
			    int64_t	  is_causal,
			    int64_t	  qk_quant_gran,
			    double	  sm_scale,
			    int64_t	  return_lse)
{
	return qk_int8_sv_f16_accum_f16_attn(query, key, value, output, query_scale, key_scale, static_cast<int>(tensor_layout), static_cast<int>(is_causal),
					     static_cast<int>(qk_quant_gran), static_cast<float>(sm_scale), static_cast<int>(return_lse));
}

torch::Tensor wrap_f16_fuse_v_mean_attn(torch::Tensor query,
					torch::Tensor key,
					torch::Tensor value,
					torch::Tensor output,
					torch::Tensor query_scale,
					torch::Tensor key_scale,
					torch::Tensor value_mean,
					int64_t	      tensor_layout,
					int64_t	      is_causal,
					int64_t	      qk_quant_gran,
					double	      sm_scale,
					int64_t	      return_lse)
{
	return qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(query, key, value, output, query_scale, key_scale, value_mean, static_cast<int>(tensor_layout),
							 static_cast<int>(is_causal), static_cast<int>(qk_quant_gran), static_cast<float>(sm_scale),
							 static_cast<int>(return_lse));
}

// 3. THE NATIVE ATEN REGISTRY: The "Tensor(a!)" tells Torch Dynamo exactly what mutates.
TORCH_LIBRARY(attn_sm80, m)
{
	m.def("qk_int8_sv_f16_accum_f32_attn(Tensor query, Tensor key, Tensor value, Tensor(a!) output, Tensor query_scale, Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
	m.def("qk_int8_sv_f16_accum_f16_attn(Tensor query, Tensor key, Tensor value, Tensor(a!) output, Tensor query_scale, Tensor key_scale, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
	m.def("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(Tensor query, Tensor key, Tensor value, Tensor(a!) output, Tensor query_scale, Tensor key_scale, Tensor value_mean, int tensor_layout, int is_causal, int qk_quant_gran, float sm_scale, int return_lse) -> Tensor");
}

// 4. THE DISPATCH BINDING: Connects the ATen registry to our 64-bit Adapters.
TORCH_LIBRARY_IMPL(attn_sm80, CUDA, m)
{
	m.impl("qk_int8_sv_f16_accum_f32_attn", &wrap_f32_attn);
	m.impl("qk_int8_sv_f16_accum_f16_attn", &wrap_f16_attn);
	m.impl("qk_int8_sv_f16_accum_f16_fuse_v_mean_attn", &wrap_f16_fuse_v_mean_attn);
}
