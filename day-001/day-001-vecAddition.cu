#include <cuda_runtime.h>

void vecAdd(float *A_h, float *B_h, float *C_h, int elements)
{
    // how many bytes do wee need to allocate?
    int size = elements * sizeof(float);

    float *A_d, *B_d, *C_d;

    // allocate memory on device for the arrays
    cudaMalloc((void **)&A_d, size);
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