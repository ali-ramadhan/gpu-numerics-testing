/* To compile for an NVIDIA P100 GPU:
 * nvcc --std=c++11 -g -G -lcusolver -lcublas -lcusparse
 * -gencode=arch=compute_30,code=sm_30
 * -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50
 * -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52
 * -I ~/NVIDIA_CUDA-7.5_Samples/common/inc/ cuda_wave_equation_1D.cu -o cuda_wave_equation_1D.out
 *
 * See: https://docs.nvidia.com/cuda/pascal-compatibility-guide/index.html
 */

#include <iostream>
#include <fstream>
#include <random>

#include <cuda_runtime.h>

#include "cublas_v2.h"
#include "cusolverDn.h"
#include "helper_cuda.h"
#include "helper_cusolver.h"

using namespace std;

/**
* Solves the linear system Ax = b using LU factorization.
* 
* Note that there are four functions: cusolverDnSgetrs for float* A, cusolverDnDgetrs for double* A,
* cusolverDnCgetrs for cuComplex* A, and cusolverDnZgetrs for cuDoubleComplex* A.
*
* @param handle Handle to the cuSolveDN library context.
* @param n Number of rows and columns of matrix A.
* @param Acopy Pointer to copy of the matrix A.
* @param lda Leading dimension of two-dimensional array used to store matrix A. 
* @param b Pointer to array containing the right hand side.
* @param x Pointer to array storing the solution x.
*
* @return The test results
*/
// __device__
int linearSolverLU(cusolverDnHandle_t handle, int n, const double *Acopy, int lda, const double *b,
                   double *x) {
    int bufferSize = 0;
    double *buffer = NULL;
    int *info = NULL;

    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence

    int h_info = 0;
    
    double start, stop;
    double time_solve;

    // Helper function that calculates the size of work buffers needed.
    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*) Acopy, lda, &bufferSize));

    // Allocate bytes of linear memory on the device.
    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));


    // prepare a copy of A because getrf will overwrite A with L.
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    start = second();
    start = second();

    /* Quick reference for cuSolverDn<t>getrf():
     * 
     * This function computes the LU factorization of a m×n matrix PA = LU where  where A is a m×n
     * matrix, P is a permutation matrix, L is a lower triangular matrix with unit diagonal, and U
     * is an upper triangular matrix.
     * 
     * @param handle Handle to the cuSolverDN library context.
     * @param m Number of rows of the matrix A.
     * @param n Number of columns of the matrix A.
     * @param nrhs Number of right hand sides.
     * @param A (device memory) Array of dimension lda * n with lda is not less than max(1,m). 
     * @param lda leading dimension of the two-dimensional array used to store matrix A.
     * @param Workspace (device memory) Working space, <type> array of size Lwork.
     * @param devIpiv (device memory) Array of size at least min(m,n), containing pivot indices.
     * @param devInfo (device memory) If devInfo = 0, the operation is successful. if devInfo = -i,
     *        the i-th parameter is wrong.
     *
     * @return status (cusolverStatus_t) one of CUSOLVER_STATUS_SUCCESS,
     *         CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_INVALID_VALUE,
     *         CUSOLVER_STATUS_ARCH_MISMATCH, or CUSOLVER_STATUS_INTERNAL_ERROR.
     */
    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: LU factorization failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    
    /* Quick reference for cusolverDn<t>getrs():
     * 
     * Solves a linear system of multiple right-hand sides op(A)*X = B where A is a n×n matrix, and
     * was LU-factored by getrf, that is, lower trianular part of A is L, and upper triangular part
     * (including diagonal elements) of A is U. B is a n×nrhs right-hand side matrix. 
     * 
     * @param handle Handle to the cuSolverDN library context.
     * @param trans Operation op(A) that is either none op(A) = A (CUBLAS_OP_N), op(A) = A^T
     *        (CUBLAS_OP_T), or op(A) = A^H (CUBLAS_OP_H).
     * @param n Number of rows and columns of the matrix A.
     * @param nrhs Number of right hand sides.
     * @param A (device memory) Array of dimension lda * n with lda is not less than max(1,n).
     * @param lda leading dimension of the two-dimensional array used to store matrix A.
     * @param devIpiv (device memory) Array of size at least n, containing pivot indices.
     * @param B (device memory) Array of dimension ldb * nrhs with ldb is not less than max(1,n).
     * @param ldb Leading dimension of two-dimensional array used to store matrix B.
     * @param devInfo (device memory) If devInfo = 0, the operation is successful. if devInfo = -i,
     *        the i-th parameter is wrong.
     *
     * @return status (cusolverStatus_t) one of CUSOLVER_STATUS_SUCCESS,
     *         CUSOLVER_STATUS_NOT_INITIALIZED, CUSOLVER_STATUS_INVALID_VALUE,
     *         CUSOLVER_STATUS_ARCH_MISMATCH, or CUSOLVER_STATUS_INTERNAL_ERROR.
     */
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));

    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();

    time_solve = stop - start;
    fprintf (stdout, "timing: LU = %10.6f sec\n", time_solve);

    if (info  ) { checkCudaErrors(cudaFree(info)); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv)); }

    return 0;
}

// __device__
double normal_pdf(double x, double m, double s) {
    const double inv_sqrt_2pi = 0.3989422804014327;
    double a = (x - m) / s;

    return (inv_sqrt_2pi / s) * std::exp(-0.5f * a * a);
}

// __device__
void solve_1D_wave_equation(double mu, double sigma, string output_filepath) {
    cusolverDnHandle_t handle = NULL;
    cublasHandle_t cublasHandle = NULL; // used in residual evaluation
    cudaStream_t stream = NULL;

    checkCudaErrors(cusolverDnCreate(&handle));
    checkCudaErrors(cublasCreate(&cublasHandle));
    checkCudaErrors(cudaStreamCreate(&stream));

    checkCudaErrors(cusolverDnSetStream(handle, stream));
    checkCudaErrors(cublasSetStream(cublasHandle, stream));

    // the constants are used in residual evaluation, r = b - A*x
    const double minus_one = -1.0;
    const double one = 1.0;

    // IOFormat CommaInitFmt(StreamPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
    // ofstream outfile(output_filepath);

    /* Problem parameters */
    double c = 1.0;  // Propagation speed of the wave.
    double L = 1.0;  // Length of the domain.
    int N = 100;     // Number of grid points.

    double dx = L/N;
    double dt = 0.01;
    double t_end = 1.0;

    double alpha = c*c * dt*dt / (2*dx*dx);

    /* We'll discretize the wave equation using the Crank-Nicolson method and
     * solve the resulting linear system A*u = b at every time step.
     */
    double *A;
    double *b;
    double *u_n;
    double *u_nm1;

    int rowsA = N;  // number of rows of A
    int colsA = N;  // number of columns of A
    int nnzA  = 0;  // number of nonzeros of A
    int baseA = 0;  // base index in CSR format
    int lda   = N;  // leading dimension in dense matrix

    checkCudaErrors(cudaMallocManaged((void **) &A, sizeof(double)*lda*colsA));
    checkCudaErrors(cudaMallocManaged((void **) &b, sizeof(double)*rowsA));
    checkCudaErrors(cudaMallocManaged((void **) &u_n, sizeof(double)*rowsA));
    checkCudaErrors(cudaMallocManaged((void **) &u_nm1, sizeof(double)*rowsA));

    // Copies of the data on device memory.
    double *d_A = NULL; // a copy of A
    double *d_x = NULL; // x = A \ b
    double *d_b = NULL; // a copy of b
    double *d_r = NULL; // r = b - A*x

    // Initialize matrix of coefficients.
    A[0]                 = 1 + 2*alpha;  // A[0][0]
    A[rowsA*(N-1) + N-1] = 1 + 2*alpha;  // A[N-1][N-1]
    A[1]                 = -alpha;       // A[0][1]
    A[rowsA*(N-1) + N-2] = -alpha;       // A[N-1][N-2]

    for (int i = 1; i < N-1; i++) {
        A[rowsA*i + i] = 1 + 2*alpha;  // A[i][i]
        A[rowsA*i + i+1] = -alpha;     // A[i][i+1]
        A[rowsA*i + i-1] = -alpha;     // A[i][i-1]
    }

    // Set initial conditions.
    u_nm1[0]   = 0;
    u_nm1[N-1] = 0;
    for (int i = 1; i < N-1; i++)
        u_nm1[i] = normal_pdf(i*dx, mu, sigma);

    // Output first row corresponding to initial condition (t = 0)
    // outfile << u_nm1.format(CommaInitFmt) << '\n';

    // Take the first time step.
    u_n[0]   = 0;
    u_n[N-1] = 0;

    for (int i = 1; i < N-1; i++)
        u_n[i] = u_nm1[i] + (c*c/2) * (u_nm1[i+1] - 2*u_nm1[i] + u_nm1[i-1]);

    // Output second row corresponding to first time step.
    // outfile << u_n.format(CommaInitFmt) << '\n';

    double t = dt; // We already took one step so t = dt now.

    while (t < t_end) {
        t += dt;

        // Set up right-hand side vector b.
        b[0]   = 2*(1-alpha)*u_n[0] - u_nm1[0] + alpha*u_n[1];
        b[N-1] = 2*(1-alpha)*u_n[N-1] - u_nm1[N-1] + alpha*u_n[N-1];
        for (int i = 1; i < N-1; i++)
            b[i] = 2*(1-alpha)*u_n[i] - u_nm1[i] + alpha*(u_n[i+1] + u_n[i-1]);

        u_nm1 = u_n;
        
        double *u_np1;
        checkCudaErrors(cudaMallocManaged((void **) &u_np1, sizeof(double)*colsA));

        checkCudaErrors(cudaMallocManaged((void **) &d_A, sizeof(double)*lda*colsA));
        checkCudaErrors(cudaMallocManaged((void **) &d_x, sizeof(double)*colsA));
        checkCudaErrors(cudaMallocManaged((void **) &d_b, sizeof(double)*rowsA));
        checkCudaErrors(cudaMallocManaged((void **) &d_r, sizeof(double)*rowsA));

        checkCudaErrors(cudaMemcpy(d_A, A, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b, b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));

        double *Acopy;
        checkCudaErrors(cudaMalloc((void **) &Acopy, sizeof(double)*lda*colsA));
        checkCudaErrors(cudaMemcpy(Acopy, A, sizeof(double)*lda*colsA, cudaMemcpyDeviceToDevice));

        linearSolverLU(handle, rowsA, Acopy, lda, b, u_np1);
        
        u_np1[0]   = 0;
        u_np1[N-1] = 0;

        // outfile << u_np1.format(CommaInitFmt) << '\n';
        
        u_n = u_np1;
    }
    
    if (handle) { checkCudaErrors(cusolverDnDestroy(handle)); }
    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (stream) { checkCudaErrors(cudaStreamDestroy(stream)); }

    if (d_A) { checkCudaErrors(cudaFree(d_A)); }
    if (d_x) { checkCudaErrors(cudaFree(d_x)); }
    if (d_b) { checkCudaErrors(cudaFree(d_b)); }
    if (d_r) { checkCudaErrors(cudaFree(d_r)); }

    // outfile.close();
}

// __global__
void solve_1D_wave_equations(int M, double *mu, double *sigma) {
    for (int i = 0; i < M; i++) {
        string file_suffix = to_string(i);
        file_suffix.insert(file_suffix.begin(), 3 - file_suffix.length(), '0');
        string filename = "cpu_wave_" + file_suffix + ".dat";

        // cout << "Solving wave equation problem " << i << "..." << " (block " << blockIdx.x
        //      << ", thread " << threadIdx.x << ", mu=" << mu[i] << ", sigma=" << sigma[i] << ")\n";
        cout << "Solving wave equation problem " << i << "..." << " (mu=" << mu[i] << ", sigma="
             << sigma[i] << ")\n";

        solve_1D_wave_equation(mu[i], sigma[i], filename);
    }
}

int main() {
    int M = 25;  // Number of 1D wave equation problems to solve.

    std::cout << "Solving " << M << " problem." << std::endl;

    std::random_device rd;  // Obtain a random number generator from hardware.
    std::mt19937 mt(rd()); // Seed the Mersenne Twister generator.

    /* We will impose a Gaussian wave initial condition for the 1D wave equation, with randomly
     * generated means (0.2 < mu < 0.8) and standard deviations (0.01 < sigma < 0.5).
     */
    std::uniform_real_distribution<double> uniform_mu(0.2, 0.8);
    std::uniform_real_distribution<double> uniform_sigma(0.01, 0.5);

    double *mu, *sigma;

    std::cout << "Generating initial conditions..." << std::endl;

    // Allocate unified memory
    cudaMallocManaged(&mu, M*sizeof(double));
    cudaMallocManaged(&sigma, M*sizeof(double));

    for(int i = 0; i < M; i++) {
        mu[i] = uniform_mu(mt);
        sigma[i] = uniform_sigma(mt);
    }

    std::cout << "Gaussian initial conditions randomly generated." << std::endl;

    // Solve the M problems on the CPU.
    // solve_1D_wave_equations<<<1, 1>>>(M, mu, sigma);
    solve_1D_wave_equations(M, mu, sigma);

    cudaDeviceSynchronize();

    cudaFree(mu);
    cudaFree(sigma);

    cudaDeviceReset();
}
