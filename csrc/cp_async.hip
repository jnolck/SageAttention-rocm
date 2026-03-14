/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * This file is based on code from Flashinfer,
 * https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/cp_async.cuh
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

#pragma once
#include <hip/hip_runtime.h>

namespace cp_async {

enum class SharedMemFillMode {
        kFillZero, // Fill zero to shared memory when predicate is false
        kNoFill    // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
        kNoPrefetch, // Do not fetch additional data from global memory to L2
        kPrefetch    // Fetch additional data from global memory to L2
};

// Take the fallback and let the compiler do it's thing

__device__ __forceinline__ void commit_group() {
        // got replaced
}

template <size_t n> __device__ __forceinline__ void wait_group() {}

template <PrefetchMode prefetch_mode, typename T> __device__ __forceinline__ void load_128b(T *smem_ptr, const T *gmem_ptr) { *((uint4 *)smem_ptr) = *((uint4 *)gmem_ptr); }

template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T> __device__ __forceinline__ void pred_load_128b(T *smem_ptr, const T *gmem_ptr, bool predicate) {
        if (predicate) {
                *((uint4 *)smem_ptr) = *((uint4 *)gmem_ptr);
        } else {
                if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
                        *((uint4 *)smem_ptr) = make_uint4(0, 0, 0, 0);
                }
        }
}

} // namespace cp_async
