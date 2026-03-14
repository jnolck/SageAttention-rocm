#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <rocwmma/rocwmma.hpp>
#include <vector>

#define CHECK_HIP(error)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   \
        if (error != hipSuccess) {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
                fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    \
                exit(EXIT_FAILURE);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
        }

// ----------------------------------------------------------------------------
// TEST 3: MATRIX B (ROW MAJOR)
// EXPECTATION:
// We need to know how the hardware loads Matrix V.
// If Matrix A split the rows (Left Half/Right Half), does Matrix B split the
// columns (Top Half/Bottom Half)? We will feed it 0-255 and inspect the VGPRs.
// ----------------------------------------------------------------------------
__global__ void test_matrix_b_row_major(const __half *in_matrix, __half *out_vgprs) {
        uint32_t          lane_id = threadIdx.x;
        __shared__ __half smem[256];

        for (int i = lane_id; i < 256; i += 32)
                smem[i] = in_matrix[i];
        __syncthreads();

        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::row_major> fragB;
        rocwmma::load_matrix_sync(fragB, smem, 16);

        // Dump VGPRs
        for (int i = 0; i < 16; i++) {
                out_vgprs[lane_id * 16 + i] = fragB.x[i];
        }
}

// ----------------------------------------------------------------------------
// TEST 4: MATRIX B (COL MAJOR)
// EXPECTATION:
// We need to know how the hardware loads Matrix K^T.
// If we tell rocWMMA the memory is col_major, does it natively transpose the
// VGPR layout for us? We will feed it 0-255 and see if the layout flips.
// ----------------------------------------------------------------------------
__global__ void test_matrix_b_col_major(const __half *in_matrix, __half *out_vgprs) {
        uint32_t          lane_id = threadIdx.x;
        __shared__ __half smem[256];

        for (int i = lane_id; i < 256; i += 32)
                smem[i] = in_matrix[i];
        __syncthreads();

        rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, __half, rocwmma::col_major> fragB;
        rocwmma::load_matrix_sync(fragB, smem, 16);

        for (int i = 0; i < 16; i++) {
                out_vgprs[lane_id * 16 + i] = fragB.x[i];
        }
}

// ----------------------------------------------------------------------------
// TEST 5: NATIVE SOFTMAX ROW-MAX
// EXPECTATION:
// Test 1 proved Row 0 is spread across Threads 0-15 in Register 0.
// To find the maximum of Row 0 without touching shared memory, Threads 0-15
// must reduce their Register 0 values together using __shfl_xor.
// We will inject a massive value at Col 13, and see if the shuffle successfully
// broadcasts that max value to all 16 threads in the row.
// ----------------------------------------------------------------------------
__global__ void test_native_row_max(float *out_maxes) {
        uint32_t lane_id = threadIdx.x;

        // Base value: Thread ID
        float my_val = (float)lane_id;

        // Inject a spike at Thread 13 (Col 13 of Row 0)
        if (lane_id == 13)
                my_val = 999.0f;

        // Perform a Half-Warp Reduction (Threads 0-15 talk, Threads 16-31 talk)
        float row_max = my_val;
        row_max       = max(row_max, __shfl_xor(row_max, 8));
        row_max       = max(row_max, __shfl_xor(row_max, 4));
        row_max       = max(row_max, __shfl_xor(row_max, 2));
        row_max       = max(row_max, __shfl_xor(row_max, 1));

        // Output the result for the CPU to verify
        out_maxes[lane_id] = row_max;
}

void print_vgprs(const std::vector<_Float16> &vgprs, int threads, int elements_per_thread) {
        for (int t = 0; t < threads; t++) {
                std::cout << "Thread " << std::setw(2) << t << " VGPRs: [ ";
                for (int e = 0; e < elements_per_thread; e++) {
                        std::cout << std::setw(4) << (float)vgprs[t * elements_per_thread + e] << " ";
                }
                std::cout << "]\n";
        }
}

int main() {
        std::cout << "=== RDNA 3 Wave32 Silicon Lab 2 ===\n\n";

        std::vector<_Float16> h_in_mat(256);
        for (int i = 0; i < 256; i++)
                h_in_mat[i] = (_Float16)i;
        __half *d_in_mat;
        __half *d_out_vgprs;
        CHECK_HIP(hipMalloc(&d_in_mat, 256 * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_out_vgprs, 32 * 16 * sizeof(__half)));
        CHECK_HIP(hipMemcpy(d_in_mat, h_in_mat.data(), 256 * sizeof(__half), hipMemcpyHostToDevice));
        std::vector<_Float16> h_out_vgprs(32 * 16);

        std::cout << "--- TEST 3: Matrix B (Row Major) ---\n";
        hipLaunchKernelGGL(test_matrix_b_row_major, dim3(1), dim3(32), 0, 0, d_in_mat, d_out_vgprs);
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out_vgprs.data(), d_out_vgprs, 32 * 16 * sizeof(__half), hipMemcpyDeviceToHost));
        print_vgprs(h_out_vgprs, 32, 16);

        std::cout << "\n--- TEST 4: Matrix B (Col Major) ---\n";
        hipLaunchKernelGGL(test_matrix_b_col_major, dim3(1), dim3(32), 0, 0, d_in_mat, d_out_vgprs);
        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipMemcpy(h_out_vgprs.data(), d_out_vgprs, 32 * 16 * sizeof(__half), hipMemcpyDeviceToHost));
        print_vgprs(h_out_vgprs, 32, 16);

        std::cout << "\n--- TEST 5: Native Row-Max Reduction ---\n";
        float *d_out_maxes;
        CHECK_HIP(hipMalloc(&d_out_maxes, 32 * sizeof(float)));
        hipLaunchKernelGGL(test_native_row_max, dim3(1), dim3(32), 0, 0, d_out_maxes);
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> h_out_maxes(32);
        CHECK_HIP(hipMemcpy(h_out_maxes.data(), d_out_maxes, 32 * sizeof(float), hipMemcpyDeviceToHost));

        std::cout << "Max values calculated by Threads 0-15 (Expected: 999): \n[ ";
        for (int i = 0; i < 16; i++)
                std::cout << h_out_maxes[i] << " ";
        std::cout << "]\n";

        hipFree(d_in_mat);
        hipFree(d_out_vgprs);
        hipFree(d_out_maxes);
        return 0;
}
