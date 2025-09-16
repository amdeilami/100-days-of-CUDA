#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void vectorCoefficientKernel(float *A_d, float *R_d, float coefficient, int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        R_d[i] = A_d[i] * coefficient;
    }
}

__host__ void vectorCoefficient(float *A_h, float *R_h, float coefficient, int n)
{
    // allocating memory on device
    int size = n * sizeof(float);

    float *A_d, *R_d;
    cudaError_t err = cudaMalloc((void **)&A_d, size);
    if (err != cudaSuccess)
    {
        printf("There was something wrong with cuda memory allocation: %s", cudaGetErrorString(err));
    }
    cudaMalloc((void **)&R_d, size);

    // transfering data from host
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    // calling the kernel...
    vectorCoefficientKernel<<<ceil(n / 256.0), 256>>>(A_d, R_d, coefficient, n);

    // transfering result to host
    cudaMemcpy(R_h, R_d, size, cudaMemcpyDeviceToHost);

    // deallocating what we don't need anymore
    cudaFree(A_d);
    cudaFree(R_d);
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    float *A_h = (float *)malloc(n * sizeof(float));
    float *R_h = (float *)malloc(n * sizeof(float));

    srand((unsigned int)time(NULL));

    float coefficient = (float)(rand() % 10000000000);

    for (int i = 0; i < n; ++i)
    {
        A_h[i] = (float)(rand() % 1444);
        R_h[i] = 0.0;
    }

    printf("coefficient: %f\n", coefficient);

    for (int i = 0; i < n; ++i)
    {
        printf("%f ,", A_h[i]);
    }

    vectorCoefficient(A_h, R_h, coefficient, n);

    printf("\n\n");

    for (int i = 0; i < n; ++i)
    {
        printf("%f ,", R_h[i]);
    }

    printf("\n\n");

    free(A_h);
    free(R_h);

    return 0;
}