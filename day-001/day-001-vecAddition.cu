#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        // we are still in the acceptable region of data
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

    // will do this later, the kernel...

    // copy the results to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // free the allocated objects
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    return 0;
}