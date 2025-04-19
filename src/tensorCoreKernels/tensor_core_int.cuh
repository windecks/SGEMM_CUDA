#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

using namespace nvcuda;

__global__ void naiveTensorCoresInt(const int8_t *A, const int8_t *B,
                                    int32_t *C, int M, int N, int K) {
  // Leading dimensions of A and B matrices
  int lda = K;
  int ldb = N;
  int ldc = N;

  // Fragment declaration
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> acc_frag;

  wmma::fill_fragment(acc_frag, 0);

  // Calculate the row and column of the C matrix to be computed by this thread
  // block which is also a warp
  int row = blockIdx.y * WMMA_M;
  int col = blockIdx.x * WMMA_N;

  // Loop over the K dimension to calculate partial results
  for (int i = 0; i < K; i += WMMA_K) {

    wmma::load_matrix_sync(a_frag, A + row * lda + i, lda);
    wmma::load_matrix_sync(b_frag, B + i * ldb + col, ldb);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  // Store the result
  wmma::store_matrix_sync(C + row * ldc + col, acc_frag, ldc,
                          wmma::mem_row_major);
}