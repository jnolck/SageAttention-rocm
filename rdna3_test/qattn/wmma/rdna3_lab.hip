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
// TEST 1: THE ACCUMULATOR DUMP
// EXPECTATION:
// We will manually inject the thread's lane_id directly into its local VGPRs.
// When rocWMMA stores this fragment to shared memory, the printed 16x16 grid
// will physically map exactly which thread owns which coordinates in the output.
// If it prints Even/Odd stripes, we know the hardware is scrambling.
// ----------------------------------------------------------------------------
__global__ void test_accumulator_layout(float *out_matrix) {
        uint32_t lane_id = threadIdx.x;

        // Allocate a raw 16x16 float accumulator
        rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> fragC;

        // Fill the fragment's raw VGPRs with the thread's ID
        // An accumulator on RDNA3 Wave32 should have 8 elements (8 * 32 threads = 256 elements = 16x16)
        for (int i = 0; i < 8; i++) {
                fragC.x[i] = (float)lane_id + ((float)i * 0.01f); // e.g., Thread 5 stores 5.00, 5.01, 5.02...
        }

        __shared__ float smem[256];

        // Let the hardware route the registers to memory
        rocwmma::store_matrix_sync(smem, fragC, 16, rocwmma::mem_row_major);
        __syncthreads();

        // Thread 0 dumps the shared memory back to global memory for the CPU to read
        if (lane_id == 0) {
                for (int i = 0; i < 256; i++) {
                        out_matrix[i] = smem[i];
                }
        }
}

// ----------------------------------------------------------------------------
// TEST 2: THE MATRIX A LOAD
// EXPECTATION:
// We will put sequential numbers (0 to 255) into shared memory.
// We will use rocWMMA to load them into a Matrix A fragment (16-bit).
// We will then dump the raw VGPRs back to global memory.
// This will tell us if Matrix A physically duplicates rows across threads
// (Thread 0 and 16 holding the same data) as the GPUOpen article claimed.
// ----------------------------------------------------------------------------
__global__ void test_matrix_a_layout(const __half *in_matrix, __half *out_vgprs) {
        uint32_t lane_id = threadIdx.x;

        __shared__ __half smem[256];

        // Threads collaboratively load the sequential 16x16 matrix into shared memory
        for (int i = lane_id; i < 256; i += 32) {
                smem[i] = in_matrix[i];
        }
        __syncthreads();

        rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, __half, rocwmma::row_major> fragA;

        // Load the sequential data into the matrix core registers
        rocwmma::load_matrix_sync(fragA, smem, 16);

        // Dump the raw VGPR contents straight to global memory so we can inspect them
        // Matrix A expects 16 elements (8 uint32_t registers) per thread.
        for (int i = 0; i < 16; i++) {
                out_vgprs[lane_id * 16 + i] = fragA.x[i];
        }

        // If you attach rocgdb here and step into this kernel, you can directly
        // inspect fragA.x to see exactly what silicon registers were populated.
}

void print_matrix(const std::vector<float> &mat, int rows, int cols) {
        for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                        std::cout << std::fixed << std::setprecision(2) << std::setw(6) << mat[r * cols + c] << " ";
                }
                std::cout << "\n";
        }
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
        std::cout << "=== RDNA 3 Wave32 Silicon Lab ===\n\n";

        // --- Run Test 1 ---
        std::cout << "--- TEST 1: Accumulator Store Mapping ---\n";
        float *d_out_acc;
        CHECK_HIP(hipMalloc(&d_out_acc, 256 * sizeof(float)));

        hipLaunchKernelGGL(test_accumulator_layout, dim3(1), dim3(32), 0, 0, d_out_acc);
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<float> h_out_acc(256);
        CHECK_HIP(hipMemcpy(h_out_acc.data(), d_out_acc, 256 * sizeof(float), hipMemcpyDeviceToHost));

        std::cout << "Resulting 16x16 Matrix (Values indicate [ThreadID].[RegisterIndex]):\n";
        print_matrix(h_out_acc, 16, 16);
        std::cout << "\n";

        // --- Run Test 2 ---
        std::cout << "--- TEST 2: Matrix A VGPR Layout ---\n";
        std::vector<_Float16> h_in_mat(256);
        for (int i = 0; i < 256; i++)
                h_in_mat[i] = (_Float16)i;

        __half *d_in_mat;
        __half *d_out_vgprs;
        CHECK_HIP(hipMalloc(&d_in_mat, 256 * sizeof(__half)));
        CHECK_HIP(hipMalloc(&d_out_vgprs, 32 * 16 * sizeof(__half)));
        CHECK_HIP(hipMemcpy(d_in_mat, h_in_mat.data(), 256 * sizeof(__half), hipMemcpyHostToDevice));

        hipLaunchKernelGGL(test_matrix_a_layout, dim3(1), dim3(32), 0, 0, d_in_mat, d_out_vgprs);
        CHECK_HIP(hipDeviceSynchronize());

        std::vector<_Float16> h_out_vgprs(32 * 16);
        CHECK_HIP(hipMemcpy(h_out_vgprs.data(), d_out_vgprs, 32 * 16 * sizeof(__half), hipMemcpyDeviceToHost));

        print_vgprs(h_out_vgprs, 32, 16);

        hipFree(d_out_acc);
        hipFree(d_in_mat);
        hipFree(d_out_vgprs);

        return 0;
}
