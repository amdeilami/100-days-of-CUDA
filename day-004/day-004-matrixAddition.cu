#include <stdio.h>
#include <time.h>

__global__ void matrixAdditionKernel(float *A_d, float *B_d, float *R_d, unsigned int m, unsigned int n)
{
}

__host__ void matrixAddition(float *A_h, float *B_h, float *R_h, unsigned int m, unsigned int n)
{

    unsigned int size = m * n * sizeof(float);

    float *A_d, *B_d, *R_d;
    cudaError_t err = cudaMalloc((void **)&A_d, size);
    if (err != cudaSuccess)
    {
        printf("issue with memory allocation in device...");
        exit(1);
    }
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&R_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // calling the kernel...

    cudaMemcpy(R_h, R_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(R_d);
}

__host__ int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("enter the dimensions of the matrices, for example: ./a.out 5 3");
    }
    unsigned int m = atoi(argv[1]);
    unsigned int n = atoi(argv[2]);

    float *A_h, *B_h, *R_h;
    unsigned int size = m * n * sizeof(float);
    A_h = (float *)malloc(size);
    B_h = (float *)malloc(size);
    R_h = (float *)malloc(size);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A_h[i * m + j] = (float)(rand() % 100000000);
            B_h[i * m + j] = (float)(rand() % 100000000);
        }
    }

    matrixAddition(A_h, B_h, R_h, m, n);
}