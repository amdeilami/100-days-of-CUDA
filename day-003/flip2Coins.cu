#include <time.h>
#include <stdio.h>

__global__ void flip2CoinsKernel(bool *coins_d, int n)
{
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i = (thread_id) * 2;

    if (i < n)
    {
        coins_d[i] = !coins_d[i];
        coins_d[i + 1] = !coins_d[i + 1];
    }
}

void flip2Coins(bool *coins_h, int n)
{

    bool *coins_d;
    cudaError_t err = cudaMalloc((void **)&coins_d, n * sizeof(bool));
    if (err != cudaSuccess)
    {
        printf("There was an issue with memory allocation at %d: %s", __LINE__, cudaGetErrorString(err));
    }

    cudaMemcpy(coins_d, coins_h, n * sizeof(bool), cudaMemcpyHostToDevice);

    // calling the kernel...
    flip2CoinsKernel<<<ceil(n / 64), 32>>>(coins_d, n);

    cudaMemcpy(coins_h, coins_d, n * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(coins_d);
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    bool *coins = (bool *)malloc(n * sizeof(bool));

    srand((unsigned int)time(NULL));

    for (int i = 0; i < n; ++i)
    {
        int random = rand();
        coins[i] = (random % 2 == 0 ? true : false);
    }

    for (int i = 0; i < n; ++i)
    {
        printf("%s ,", (coins[i] == true ? "T" : "H"));
    }

    flip2Coins(coins, n);

    printf("\n\n");
    for (int i = 0; i < n; ++i)
    {
        printf("%s ,", (coins[i] == true ? "T" : "H"));
    }

    printf("\n");

    return 0;
}