#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <array>

__device__ const char *charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
__device__ const int charset_sz = 62;

__host__ float crack( char *secret, char *result );
