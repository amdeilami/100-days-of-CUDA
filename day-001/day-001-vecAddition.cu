#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // see if we are still in an acceptable region of data
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(float *A_h, float *B_h, float *C_h, int elements)
{
    // how many bytes do wee need to allocate?
    int size = elements * sizeof(float);

    float *A_d, *B_d, *C_d;

    // a sample of error handling for CUDA functions
    cudaError_t err = cudaMalloc((void **)&A_d, size);

    if (err != cudaSuccess)
    {
        printf("There was an issue in memory allocation above line %d: %s", __LINE__, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // allocate memory on device for the arrays (for vector A, I also have error handling sample)
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // copy from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // call the kernel
    vecAddKernel<<<ceil(elements / 256.0), 256>>>(A_d, B_d, C_d, elements);

    // copy the results to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free the allocated objects
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main(int argc, char *argv[])
{
    // getting the dimension of vector from user
    int n = atoi(argv[1]);
    int size = n * sizeof(float);

    float *A_h = (float *)malloc(size);
    float *B_h = (float *)malloc(size);
    float *C_h = (float *)malloc(size);

    srand((unsigned int)time(NULL));

    for (int i = 0; i < n; ++i)
    {
        A_h[i] = (float)rand() / RAND_MAX;
        B_h[i] = (float)rand() / RAND_MAX;
        C_h[i] = 0.0;
    }

    vecAdd(A_h, B_h, C_h, n);

    printf("\n");
    for (int i = 0; i < n; i++)
    {
        printf("%f , ", C_h[i]);
    }
    printf("\n");

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}