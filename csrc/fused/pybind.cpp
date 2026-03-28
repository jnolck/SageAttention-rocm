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
#include "fused.h"
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// 1. THE TROJAN HORSE: Keeps Python's import system happy
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
}

// 2. THE 64-BIT ADAPTERS: PyTorch 2.10 demands int64_t and double at the boundary.
// Notice we are returning void to match your original fused.h functions.

void wrap_quant_per_block_int8_cuda_sm(torch::Tensor input,
				       torch::Tensor output,
				       torch::Tensor scale,
				       double	     sm_scale,
				       int64_t	     block_size,
				       int64_t	     tensor_layout)
{
	quant_per_block_int8_cuda(input, output, scale, static_cast<float>(sm_scale), static_cast<int>(block_size), static_cast<int>(tensor_layout));
}

void wrap_quant_per_block_int8_cuda(torch::Tensor input, torch::Tensor output, torch::Tensor scale, int64_t block_size, int64_t tensor_layout)
{
	quant_per_block_int8_cuda(input, output, scale, static_cast<int>(block_size), static_cast<int>(tensor_layout));
}

void wrap_quant_per_block_int8_fuse_sub_mean_cuda(torch::Tensor input,
						  torch::Tensor mean,
						  torch::Tensor output,
						  torch::Tensor scale,
						  int64_t	block_size,
						  int64_t	tensor_layout)
{
	quant_per_block_int8_fuse_sub_mean_cuda(input, mean, output, scale, static_cast<int>(block_size), static_cast<int>(tensor_layout));
}

void wrap_quant_per_warp_int8_cuda(torch::Tensor input,
				   torch::Tensor output,
				   torch::Tensor scale,
				   int64_t	 block_size,
				   int64_t	 warp_block_size,
				   int64_t	 tensor_layout)
{
	quant_per_warp_int8_cuda(input, output, scale, static_cast<int>(block_size), static_cast<int>(warp_block_size), static_cast<int>(tensor_layout));
}

void wrap_sub_mean_cuda(torch::Tensor input, torch::Tensor mean, torch::Tensor output, int64_t tensor_layout)
{
	sub_mean_cuda(input, mean, output, static_cast<int>(tensor_layout));
}

void wrap_transpose_pad_permute_cuda(torch::Tensor input, torch::Tensor output, int64_t tensor_layout)
{
	transpose_pad_permute_cuda(input, output, static_cast<int>(tensor_layout));
}

void wrap_scale_fuse_quant_cuda(torch::Tensor input, torch::Tensor output, torch::Tensor scale, int64_t num_tokens, double scale_max, int64_t tensor_layout)
{
	scale_fuse_quant_cuda(input, output, scale, static_cast<int>(num_tokens), static_cast<float>(scale_max), static_cast<int>(tensor_layout));
}

void wrap_mean_scale_fuse_quant_cuda(torch::Tensor input,
				     torch::Tensor output,
				     torch::Tensor mean,
				     torch::Tensor scale,
				     int64_t	   num_tokens,
				     double	   scale_max,
				     int64_t	   tensor_layout)
{
	mean_scale_fuse_quant_cuda(input, output, mean, scale, static_cast<int>(num_tokens), static_cast<float>(scale_max), static_cast<int>(tensor_layout));
}

// 3. THE ATEN REGISTRY: The "Tensor(a!)" tells Torch Dynamo exactly what memory mutates.
TORCH_LIBRARY(sage_fused_quant, m)
{
	m.def("quant_per_block_int8_cuda_sm(Tensor input, Tensor(a!) output, Tensor(b!) scale, float sm_scale, int block_size, int tensor_layout) -> ()");
	m.def("quant_per_block_int8_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, int block_size, int tensor_layout) -> ()");
	m.def("quant_per_block_int8_fuse_sub_mean_cuda(Tensor input, Tensor(a!) mean, Tensor(b!) output, Tensor(c!) scale, int block_size, int tensor_layout) -> ()");
	m.def("quant_per_warp_int8_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, int block_size, int warp_block_size, int tensor_layout) -> ()");
	m.def("sub_mean_cuda(Tensor input, Tensor(a!) mean, Tensor(b!) output, int tensor_layout) -> ()");
	m.def("transpose_pad_permute_cuda(Tensor input, Tensor(a!) output, int tensor_layout) -> ()");
	m.def("scale_fuse_quant_cuda(Tensor input, Tensor(a!) output, Tensor(b!) scale, int num_tokens, float scale_max, int tensor_layout) -> ()");
	m.def("mean_scale_fuse_quant_cuda(Tensor input, Tensor(a!) output, Tensor(b!) mean, Tensor(c!) scale, int num_tokens, float scale_max, int tensor_layout) -> ()");
}

// 4. THE DISPATCH BINDING: Connects the ATen registry to our 64-bit Adapters.
TORCH_LIBRARY_IMPL(sage_fused_quant, CUDA, m)
{
	m.impl("quant_per_block_int8_cuda_sm", &wrap_quant_per_block_int8_cuda_sm);
	m.impl("quant_per_block_int8_cuda", &wrap_quant_per_block_int8_cuda);
	m.impl("quant_per_block_int8_fuse_sub_mean_cuda", &wrap_quant_per_block_int8_fuse_sub_mean_cuda);
	m.impl("quant_per_warp_int8_cuda", &wrap_quant_per_warp_int8_cuda);
	m.impl("sub_mean_cuda", &wrap_sub_mean_cuda);
	m.impl("transpose_pad_permute_cuda", &wrap_transpose_pad_permute_cuda);
	m.impl("scale_fuse_quant_cuda", &wrap_scale_fuse_quant_cuda);
	m.impl("mean_scale_fuse_quant_cuda", &wrap_mean_scale_fuse_quant_cuda);
}
