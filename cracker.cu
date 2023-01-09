#include "cracker.cuh"
#include "md5.cuh"

using namespace std;
extern __shared__ char s[];
typedef unsigned long long ull;
// const unsigned int m = 1 << 30, n = 62;
const unsigned int m = 128, n = 64;
// const unsigned int m = 32, n = 62;

__global__ void thread_attempt( char *secret, char *result, bool *found )
{
    const unsigned tid = threadIdx.x;

    // Load secret hash & charset into shared memory
    if (tid < word) s[tid] = secret[tid];
    if (tid < charset_sz) s[word + tid] = charset[tid];
    
    __syncthreads();
    
    // We check result before writing to avoid race condition
    unsigned count = 0;
    bool res;
    do
    {
        res = MD5(s, &count);
    } while (res == 0 && *found == 0);

    if (res)
    {
        // Used to signal to the remaining threads that the password has been found
        *found = true;
        // Copy the password to the result
        memcpy(result, s, word);
    }

}

#include "md5.h"

// Returns the password 
__host__ float crack( char *secret, char *deciphered )
{
    float delta;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    char *data, *result;
    cudaMalloc((void**)&data, sizeof(char) * MD5::size);
    cudaMalloc((void**)&result, sizeof(char) * MD5::size);
    cudaMemcpy(data, secret, sizeof(char) * MD5::size, cudaMemcpyHostToDevice);

    bool *found;
    cudaMalloc((void**) &found, sizeof(bool));

    // Start the timer
    cudaEventRecord(start);
    
    // Run the cracking algorithm
    thread_attempt<<<m, n, MD5::hash_sz + charset_sz * sizeof(char) + sizeof(ull)>>>(data, result, found);
    cudaDeviceSynchronize();

    // Stop the timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate the time delta
    cudaEventElapsedTime(&delta, start, stop);

    bool status;
    cudaMemcpy(&status, found, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(found);

    cudaMemcpy(deciphered, result, sizeof(char) * MD5::size, cudaMemcpyDeviceToHost);
    
    cudaFree(data);
    cudaFree(result);

    return delta;
}
