#include "cuda_fp16.h"
#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}
void initialize_one_hf(__half *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = __float2half(1.0);
  }
}
void initialize_incremental_hf(__half *mat, int N) {
  for (int i = 0; i < N; i++) {
    int tmp = i / 64;
    mat[i] = static_cast<half>(tmp);
  }
}
void initialize_incremental_int8(int8_t *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}
void initialize_identity_hf(__half *mat, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == j) {
        mat[i * N + j] = __float2half(1.0);
      } else {
        mat[i * N + j] = __float2half(0.0);
      }
    }
  }
}
void initialize_identity_int8(int8_t *mat, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (i == j) {
        mat[i * N + j] = 1;
      } else {
        mat[i * N + j] = 0;
      }
    }
  }
}
void initialize_incremental_float(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = (float)(i);
  }
}
void initialize_one_float(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 1.0;
  }
}
void initialize_one_int8(int8_t *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 1;
  }
}
void initialize_one_int(int32_t *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 1;
  }
}
void randomize_matrix_hf(__half *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = __float2half(tmp);
  }
}

void randomize_matrix_int8(int8_t *mat, int N) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = int8_t(tmp);
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void zero_init_matrix_int8(int8_t *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void print_matrix_hf(const __half *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    // Convert __half to float before streaming
    float val = __half2float(A[i]);
    if ((i + 1) % N == 0)
      fs << std::setw(5) << val; // Set field width and write the value
    else
      fs << std::setw(5) << val << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void print_matrix_int8(const int8_t *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << (int)A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << (int)A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void print_matrix_int(const int32_t *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

void print_matrix_int_transposed(const int32_t *A, int M, int N,
                                 std::ofstream &fs) {
  fs << "[";
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      int idx = col * M + row; // because A is transposed
      if (col == N - 1) {
        fs << std::setw(5) << A[idx]; // last column, no comma
      } else {
        fs << std::setw(5) << A[idx] << ", ";
      }
    }
    if (row != M - 1) {
      fs << ";\n"; // semicolon + newline after each row except last
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

bool verify_matrix_hf(__half *matRef, __half *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(__half2float(matRef[i] - matOut[i]));
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             __half2float(matRef[i]), __half2float(matOut[i]), diff, i);
      return false;
    }
  }
  return true;
}

bool verify_matrix_int(int32_t *matRef, int32_t *matOut, int m, int n) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      int ref_idx =
          col * m + row; // matRef is transposed version: (col, row) layout
      int out_idx = row * n + col; // matOut is normal: (row, col) layout
      int diff = std::abs(matRef[ref_idx] - matOut[out_idx]);
      if (diff != 0) {
        printf("Divergence! Should %d, Is %d (Diff %d) at (row=%d, col=%d)\n",
               matRef[ref_idx], matOut[out_idx], diff, row, col);
        return false;
      }
    }
  }
  return true;
}

void float_array_to_half(__half *half_mat, float *float_mat, int size) {
  int i;
  for (i = 0; i < size; i++) {
    half_mat[i] = __float2half(float_mat[i]);
  }
}

void half_array_to_float(__half *half_mat, float *float_mat, int size) {
  int i;
  for (i = 0; i < size; i++) {
    float_mat[i] = __half2float(half_mat[i]);
  }
}

// void float_array_to_int(int32_t *int_mat, float *float_mat, int size) {
//   int i;
//   for (i = 0; i < size; i++) {
//     int_mat[i] = __float2int_rn(float_mat[i]);
//   }
// }
//
// void int_array_to_float(int32_t *int_mat, float *float_mat, int size) {
//   int i;
//   for (i = 0; i < size; i++) {
//     float_mat[i] = __int2float_rn(int_mat[i]);
//   }
// }
//
// void half_array_to_int(int32_t *int_mat, __half *half_mat, int size) {
//   int i;
//   for (i = 0; i < size; i++) {
//     int_mat[i] = __half2int_rn(half_mat[i]);
//   }
// }
//
// void int_array_to_half(int32_t *int_mat, __half *half_mat, int size) {
//   int i;
//   for (i = 0; i < size; i++) {
//     half_mat[i] = __int2half_rn(int_mat[i]);
//   }
// }

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major order. So we change the order of our row-major A
  // & B, since (B^T*A^T)^T = (A*B) This runs cuBLAS in full fp32 mode
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasINT8(cublasHandle_t handle, int M, int N, int K, int32_t alpha,
                   int8_t *A, int8_t *B, int32_t beta, int32_t *C) {
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha, A, CUDA_R_8I,
               M, B, CUDA_R_8I, K, &beta, C, CUDA_R_32I, M,
               CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasBF16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasFP16(cublasHandle_t handle, int M, int N, int K, float alpha,
                   __half *A, __half *B, float beta, float *C) {

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
               N, A, CUDA_R_16F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void runCublasTF32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // This runs cuBLAS with mixed precision (performing the mul with operands
  // downcast to bf16), which is ~4x faster
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N,
               CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // L1 cache becomes useless, since we access GMEM only via SMEM, so we carve
  // out all of L1 to SMEM. This doesn't currently make a difference, since
  // occupancy is limited by reg and thread count, but it's good to do anyway.
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankConflicts(int M, int N, int K, float alpha, float *A,
                                  float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankConflicts<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmResolveBankExtraCol(int M, int N, int K, float alpha, float *A,
                                 float *B, float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmResolveBankExtraCol<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // A100
  // const uint K9_BK = 16;
  // const uint K9_TM = 4;
  // const uint K9_TN = 4;
  // const uint K9_BM = 64;
  // const uint K9_BN = 64;
  // A6000
  const uint K9_BK = 16;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);

  static_assert(
      (K9_NUM_THREADS * 4) % K9_BK == 0,
      "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of Bs "
      "during each iteraion)");
  static_assert(
      (K9_NUM_THREADS * 4) % K9_BN == 0,
      "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization issues "
      "during GMEM->SMEM tiling (loading only parts of the final row of As "
      "during each iteration)");
  static_assert(
      K9_BN % (16 * K9_TN) == 0,
      "K9_BN must be a multiple of 16*K9_TN to avoid quantization effects");
  static_assert(
      K9_BM % (16 * K9_TM) == 0,
      "K9_BM must be a multiple of 16*K9_TM to avoid quantization effects");
  static_assert((K9_BM * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BM*K9_BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K9_BN * K9_BK) % (4 * K9_NUM_THREADS) == 0,
                "K9_BN*K9_BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // Settings for A100
  // const uint K10_NUM_THREADS = 128;
  // const uint K10_BN = 128;
  // const uint K10_BM = 64;
  // const uint K10_BK = 16;
  // const uint K10_WN = 64;
  // const uint K10_WM = 32;
  // const uint K10_WNITER = 1;
  // const uint K10_TN = 4;
  // const uint K10_TM = 4;
  // Settings for A6000
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 128;
  const uint K10_BM = 128;
  const uint K10_BK = 16;
  const uint K10_WN = 64;
  const uint K10_WM = 64;
  const uint K10_WNITER = 4;
  const uint K10_TN = 4;
  const uint K10_TM = 8;
  dim3 blockDim(K10_NUM_THREADS);

  constexpr uint NUM_WARPS = K10_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K10_BN % K10_WN == 0) and (K10_BM % K10_WM == 0));
  static_assert((K10_BN / K10_WN) * (K10_BM / K10_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K10_WM * K10_WN) % (WARPSIZE * K10_TM * K10_TN * K10_WNITER) ==
                0);
  constexpr uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);
  // warpsubtile in warptile
  static_assert((K10_WM % K10_WMITER == 0) and (K10_WN % K10_WNITER == 0));

  static_assert((K10_NUM_THREADS * 4) % K10_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K10_NUM_THREADS * 4) % K10_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K10_BN % (16 * K10_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K10_BM % (16 * K10_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K10_BM * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K10_BN * K10_BK) % (4 * K10_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering(int M, int N, int K, float alpha, float *A,
                             float *B, float beta, float *C) {
  // Settings for A100
  // const uint K11_NUM_THREADS = 256;
  // const uint K11_BN = 128;
  // const uint K11_BM = 64;
  // const uint K11_BK = 16;
  // const uint K11_WN = 32;
  // const uint K11_WM = 32;
  // const uint K11_WNITER = 2;
  // const uint K11_TN = 4;
  // const uint K11_TM = 4;
  // Settings for A6000
  const uint K11_NUM_THREADS = 256;
  const uint K11_BN = 256;
  const uint K11_BM = 128;
  const uint K11_BK = 16;
  const uint K11_WN = 32;
  const uint K11_WM = 128;
  const uint K11_WNITER = 1;
  const uint K11_TN = 8;
  const uint K11_TM = 8;
  dim3 blockDim(K11_NUM_THREADS);

  constexpr uint NUM_WARPS = K11_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K11_BN % K11_WN == 0) and (K11_BM % K11_WM == 0));
  static_assert((K11_BN / K11_WN) * (K11_BM / K11_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K11_WM * K11_WN) % (WARPSIZE * K11_TM * K11_TN * K11_WNITER) ==
                0);
  constexpr uint K11_WMITER =
      (K11_WM * K11_WN) / (32 * K11_TM * K11_TN * K11_WNITER);
  // warpsubtile in warptile
  static_assert((K11_WM % K11_WMITER == 0) and (K11_WN % K11_WNITER == 0));

  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BK == 0,
                "NUM_THREADS*4 must be multiple of BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K11_NUM_THREADS / 2 * 4) % K11_BN == 0,
                "NUM_THREADS*4 must be multiple of BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K11_BN % (16 * K11_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K11_BM % (16 * K11_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K11_BM * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K11_BN * K11_BK) % (4 * K11_NUM_THREADS / 2) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K11_BN), CEIL_DIV(M, K11_BM));
  sgemmDoubleBuffering<K11_BM, K11_BN, K11_BK, K11_WM, K11_WN, K11_WNITER,
                       K11_TM, K11_TN, K11_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmDoubleBuffering2(int M, int N, int K, float alpha, float *A,
                              float *B, float beta, float *C) {
  // Settings for A6000
  const uint K12_NUM_THREADS = 128;
  const uint K12_BN = 128;
  const uint K12_BM = 128;
  const uint K12_BK = 16;
  const uint K12_WN = 64;
  const uint K12_WM = 64;
  const uint K12_WNITER = 4;
  const uint K12_TN = 4;
  const uint K12_TM = 8;
  dim3 blockDim(K12_NUM_THREADS);

  constexpr uint NUM_WARPS = K12_NUM_THREADS / 32;

  // warptile in threadblocktile
  static_assert((K12_BN % K12_WN == 0) and (K12_BM % K12_WM == 0));
  static_assert((K12_BN / K12_WN) * (K12_BM / K12_WM) == NUM_WARPS);

  // threads in warpsubtile
  static_assert((K12_WM * K12_WN) % (WARPSIZE * K12_TM * K12_TN * K12_WNITER) ==
                0);
  constexpr uint K12_WMITER =
      (K12_WM * K12_WN) / (32 * K12_TM * K12_TN * K12_WNITER);
  // warpsubtile in warptile
  static_assert((K12_WM % K12_WMITER == 0) and (K12_WN % K12_WNITER == 0));

  static_assert((K12_NUM_THREADS * 4) % K12_BK == 0,
                "NUM_THREADS*4 must be multiple of K9_BK to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of Bs during each iteraion)");
  static_assert((K12_NUM_THREADS * 4) % K12_BN == 0,
                "NUM_THREADS*4 must be multiple of K9_BN to avoid quantization "
                "issues during GMEM->SMEM tiling (loading only parts of the "
                "final row of As during each iteration)");
  static_assert(K12_BN % (16 * K12_TN) == 0,
                "BN must be a multiple of 16*TN to avoid quantization effects");
  static_assert(K12_BM % (16 * K12_TM) == 0,
                "BM must be a multiple of 16*TM to avoid quantization effects");
  static_assert((K12_BM * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BM*BK must be a multiple of 4*256 to vectorize loads");
  static_assert((K12_BN * K12_BK) % (4 * K12_NUM_THREADS) == 0,
                "BN*BK must be a multiple of 4*256 to vectorize loads");

  dim3 gridDim(CEIL_DIV(N, K12_BN), CEIL_DIV(M, K12_BM));
  runSgemmDoubleBuffering2<K12_BM, K12_BN, K12_BK, K12_WM, K12_WN, K12_WNITER,
                           K12_TM, K12_TN, K12_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void runSgemmIntTensorCore(int M, int N, int K, int32_t alpha, int8_t *A,
                           int8_t *B, int32_t beta, int32_t *C) {
  dim3 blockDim(32, 1, 1); // A warp per block
  dim3 gridDim(N / WMMA_N, M / WMMA_M, 1);
  runSgemmIntPtxMma<BM, BN, BK><<<gridDim, blockDim>>>(A, B, C, M, N, K);
}

void runSgemmIntTensorCoreMma(int M, int N, int K, int32_t alpha, int8_t *A,
                              int8_t *B, int32_t beta, int32_t *C) {
  const uint BK = 16;
  const uint BM = 128;
  const uint BN = 128;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((16 * BM * BN) / (WMMA_M * WMMA_N));
  runSgemmIntPtxMma<BM, BN, BK>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 7:
    runSgemmResolveBankConflicts(M, N, K, alpha, A, B, beta, C);
    break;
  case 8:
    runSgemmResolveBankExtraCol(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 11:
    runSgemmDoubleBuffering(M, N, K, alpha, A, B, beta, C);
    break;
  case 12:
    runSgemmDoubleBuffering2(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}

void run_tensor_core_kernel(int kernel_num, int M, int N, int K, float alpha,
                            __half *A, __half *B, float beta, float *C,
                            cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP16(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 13:
    throw std::invalid_argument(
        "tensor_core_int_mma is run directly in sgemm.cu.");
    break;
  case 14:
    throw std::invalid_argument("cuBLAS INT8 gemm");
  case 15:
    throw std::invalid_argument(
        "tensor_core_int_mma is run directly in sgemm.cu.");
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}