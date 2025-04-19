#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Please select a kernel (range 0 - 12, 0 for NVIDIA cuBLAS)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > 15) {
    std::cerr << "Please enter a valid kernel number (0-12)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  // print some device info
  // CudaDeviceInfo();

  // Declare the handle, create the handle, cublasCreate will return a value of
  // type cublasStatus_t to determine whether the handle was created
  // successfully (the value is 0)
  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  };
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // cuBLAS FLOPs ceiling is reached at 8192
  std::vector<int> SIZE = {128, 256, 512, 1024, 2048, 4096};
  // std::vector<int> SIZE = {256, 512,1024,2048, 4096, 8192, 16384};
  long m, n, k, max_size;
  // max_size = SIZE[SIZE.size() - 1];
  max_size = 8192;
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 1.0, beta = 0.0; // GEMM input parameters, C=α*AB+β*C
  if (kernel_num < 13 && kernel_num > 0) {
    float *A = nullptr, *B = nullptr, *C = nullptr,
          *C_ref = nullptr; // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr,
          *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * max_size * max_size);
    B = (float *)malloc(sizeof(float) * max_size * max_size);
    C = (float *)malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * max_size * max_size));
    cudaCheck(
        cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    int repeat_times = 50;
    for (int size : SIZE) {
      m = n = k = size;

      std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
      // Verify the correctness of the calculation, and execute it once before
      // the kernel function timing to avoid cold start errors
      if (kernel_num != 0) {
        run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref,
                   handle); // cuBLAS
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC,
                   handle); // Executes the kernel, modifies the result matrix
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(
            cudaGetLastError()); // Check for async errors during kernel run
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n,
                   cudaMemcpyDeviceToHost);

        if (!verify_matrix(C_ref, C, m * n)) {
          std::cout
              << "Failed to pass the correctness verification against NVIDIA "
                 "cuBLAS."
              << std::endl;
          if (m <= 256) {
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile);
            fs << "A:\n";
            print_matrix(A, m, n, fs);
            fs << "B:\n";
            print_matrix(B, m, n, fs);
            fs << "C:\n";
            print_matrix(C, m, n, fs);
            fs << "Should:\n";
            print_matrix(C_ref, m, n, fs);
          }
          exit(EXIT_FAILURE);
        }
      }

      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      long flops = 2 * m * n * k;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
      // make dC and dC_ref equal again (we modified dC while calling our kernel
      // for benchmarking)
      cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                           cudaMemcpyDeviceToDevice));
    }

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);
  }
  if (kernel_num == 13) {
    // int case. we need to run the kernel directly since run_tensor_core_kernel
    int8_t *dA = nullptr, *dB = nullptr;
    int32_t *dC = nullptr, *dC_ref = nullptr;
    auto *A = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *B = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *C = (int32_t *)malloc(sizeof(int32_t) * max_size * max_size);
    auto *C_ref = (int32_t *)malloc(sizeof(int32_t) * max_size * max_size);
    randomize_matrix_int8(A, max_size * max_size);
    zero_init_matrix_int8(B, max_size * max_size);
    B[0] = 1;
    B[129] = 1;
    B[258] = 1;
    initialize_one_int(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(int32_t) * max_size * max_size));
    cudaCheck(
        cudaMalloc((void **)&dC_ref, sizeof(int32_t) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(int32_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(int32_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    int repeat_times = 50;
    for (int size : SIZE) {
      m = n = k = size;

      std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
      // Verify the correctness of the calculation, and execute it once before
      // the kernel function timing to avoid cold start errors
      int32_t alpha_int = 1;
      int32_t beta_int = 0;
      runCublasINT8(handle, m, n, k, alpha_int, dA, dB, beta_int, dC_ref);
      runSgemmIntTensorCoreMma(m, n, k, alpha_int, dA, dB, beta_int, dC);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(int32_t) * m * n,
                 cudaMemcpyDeviceToHost);

      if (!verify_matrix_int(C_ref, C, m, n)) {
        std::cout << "Failed to pass the correctness verification against "
                     "NVIDIA cuBLAS."
                  << std::endl;
        if (m <= 4096) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix_int8(A, m, n, fs);
          fs << "B:\n";
          print_matrix_int8(B, m, n, fs);
          fs << "C:\n";
          print_matrix_int(C, m, n, fs);
          fs << "Should:\n";
          print_matrix_int_transposed(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
      for (int j = 0; j < 1000; j++) {
        // We don't reset dC between runs to save time
        runSgemmIntTensorCoreMma(m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        runSgemmIntTensorCoreMma(m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      long flops = 2 * m * n * k;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
      // make dC and dC_ref equal again (we modified dC while calling our kernel
      // for benchmarking)
      // cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
      //                      cudaMemcpyDeviceToDevice));
    }

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);
  } else if (kernel_num == 14) {
    // int8 cublas. since we aren't comparing with any other kernel, we can
    // directly run the kernel without transposing.
    int8_t *dA = nullptr, *dB = nullptr;
    int32_t *dC = nullptr;
    auto *A = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *B = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *C = (int32_t *)malloc(sizeof(int32_t) * max_size * max_size);
    randomize_matrix_int8(A, max_size * max_size);
    randomize_matrix_int8(B, max_size * max_size);
    initialize_one_int(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(int32_t) * max_size * max_size));
    cudaCheck(cudaMemcpy(dA, A, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(int32_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    int repeat_times = 50;
    for (int size : SIZE) {
      m = n = k = size;

      std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
      // Verify the correctness of the calculation, and execute it once before
      // the kernel function timing to avoid cold start errors
      int32_t alpha_int = 1;
      int32_t beta_int = 0;
      runCublasINT8(handle, m, n, k, alpha_int, dA, dB, beta_int, dC);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
      for (int j = 0; j < 1000; j++) {
        // We don't reset dC between runs to save time
        runCublasINT8(handle, m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        runCublasINT8(handle, m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      long flops = 2 * m * n * k;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
    }

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);

  } else if (kernel_num == 15) {
    // int case. we need to run the kernel directly since run_tensor_core_kernel
    int8_t *dA = nullptr, *dB = nullptr;
    int32_t *dC = nullptr, *dC_ref = nullptr;
    auto *A = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *B = (int8_t *)malloc(sizeof(int8_t) * max_size * max_size);
    auto *C = (int32_t *)malloc(sizeof(int32_t) * max_size * max_size);
    auto *C_ref = (int32_t *)malloc(sizeof(int32_t) * max_size * max_size);
    randomize_matrix_int8(A, max_size * max_size);
    zero_init_matrix_int8(B, max_size * max_size);
    B[0] = 1;
    B[129] = 1;
    B[258] = 1;
    initialize_one_int(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(int8_t) * max_size * max_size));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(int32_t) * max_size * max_size));
    cudaCheck(
        cudaMalloc((void **)&dC_ref, sizeof(int32_t) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(int8_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(int32_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(int32_t) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    int repeat_times = 50;
    for (int size : SIZE) {
      m = n = k = size;

      std::cout << "dimensions(m=n=k) " << m << ", alpha: " << alpha
                << ", beta: " << beta << std::endl;
      // Verify the correctness of the calculation, and execute it once before
      // the kernel function timing to avoid cold start errors
      int32_t alpha_int = 1;
      int32_t beta_int = 0;
      runCublasINT8(handle, m, n, k, alpha_int, dA, dB, beta_int, dC_ref);
      runSgemmIntTensorCore(m, n, k, alpha_int, dA, dB, beta_int, dC);

      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
      cudaMemcpy(C, dC, sizeof(int32_t) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(int32_t) * m * n,
                 cudaMemcpyDeviceToHost);

      if (!verify_matrix_int(C_ref, C, m, n)) {
        std::cout << "Failed to pass the correctness verification against "
                     "NVIDIA cuBLAS."
                  << std::endl;
        if (m <= 4096) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile);
          fs << "A:\n";
          print_matrix_int8(A, m, n, fs);
          fs << "B:\n";
          print_matrix_int8(B, m, n, fs);
          fs << "C:\n";
          print_matrix_int(C, m, n, fs);
          fs << "Should:\n";
          print_matrix_int_transposed(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
      for (int j = 0; j < 1000; j++) {
        // We don't reset dC between runs to save time
        runSgemmIntTensorCore(m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(beg);
      for (int j = 0; j < repeat_times; j++) {
        // We don't reset dC between runs to save time
        runSgemmIntTensorCore(m, n, k, alpha_int, dA, dB, beta_int, dC);
      }
      cudaEventRecord(end);
      cudaEventSynchronize(beg);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, beg, end);
      elapsed_time /= 1000.; // Convert to seconds

      long flops = 2 * m * n * k;
      printf(
          "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: "
          "(%ld).\n",
          elapsed_time / repeat_times,
          (repeat_times * flops * 1e-9) / elapsed_time, m);
      fflush(stdout);
    }
  }
  return 0;
};