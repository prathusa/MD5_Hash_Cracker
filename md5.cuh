#ifndef MD5_CUH
#define MD5_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <string>

typedef unsigned char byte; // 8bit
typedef unsigned int dword; // 32bit
typedef unsigned long long qword; // 64bit

__device__ const static unsigned int block_sz = 64;
__device__ const static unsigned int hash_sz = 128;
__device__ const static unsigned int word = hash_sz / (8 * sizeof(byte)); // length of the hash in bytes

// inline operations
__device__ static inline dword bm64(dword x) { return x >> 3 & 0x7f; }; // dword bytes mod 64
__device__ static inline dword F(dword x, dword y, dword z) { return (x & y) | (~x & z); };
__device__ static inline dword G(dword x, dword y, dword z) { return (x & z) | (y & ~z); };
__device__ static inline dword H(dword x, dword y, dword z) { return x ^ y ^ z; };
__device__ static inline dword I(dword x, dword y, dword z) { return y ^ (x | ~z); }
__device__ static inline dword rotate_left(dword x, int n) { return (x << n) | (x >> (32 - n)); };
__device__ static inline void FF(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + F(b, c, d) + x + ac, s) + b; };
__device__ static inline void GG(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + G(b, c, d) + x + ac, s) + b; };
__device__ static inline void HH(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + H(b, c, d) + x + ac, s) + b; };
__device__ static inline void II(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + I(b, c, d) + x + ac, s) + b; };

// Constants from https://www.rfc-editor.org/rfc/rfc1321.
const int S11 = 7, S12 = 12, S13 = 17, S14 = 22,
			S21 = 5, S22 = 9, S23 = 14, S24 = 20,
			S31 = 4, S32 = 11, S33 = 16, S34 = 23,
			S41 = 6, S42 = 10, S43 = 15, S44 = 21;

// Decodes input (byte) into output (dword). Assumes len is a multiple of 4.
__device__ void decode(dword output[], const byte input[], size_t len)
{
	for (unsigned int i = 0, j = 0; j < len; i++, j += 4)
		output[i] = ((dword)input[j]) | (((dword)input[j + 1]) << 8) | (((dword)input[j + 2]) << 16) | (((dword)input[j + 3]) << 24);
}

// Encodes input (dword) into output (byte). Assumes len is a multiple of 4.
__device__ void encode(byte output[], const dword input[], size_t len)
{
	for (size_t i = 0, j = 0; j < len; i++, j += 4)
	{
		for (int k = 0; k < 4; k++)
			output[j + k] = (input[i] >> (k * 8)) & 0xff;
	}
}

// Pretty much same as https://www.ietf.org/rfc/rfc1321.txt.
__device__ void transform(const byte block[block_sz], dword *state)
{
	dword a = state[0], b = state[1], c = state[2], d = state[3], x[16];
	decode(x, block, block_sz);

	/* Round 1 */
	FF(a, b, c, d, x[ 0], S11, 0xd76aa478); /* 1 */
	FF(d, a, b, c, x[ 1], S12, 0xe8c7b756); /* 2 */
	FF(c, d, a, b, x[ 2], S13, 0x242070db); /* 3 */
	FF(b, c, d, a, x[ 3], S14, 0xc1bdceee); /* 4 */
	FF(a, b, c, d, x[ 4], S11, 0xf57c0faf); /* 5 */
	FF(d, a, b, c, x[ 5], S12, 0x4787c62a); /* 6 */
	FF(c, d, a, b, x[ 6], S13, 0xa8304613); /* 7 */
	FF(b, c, d, a, x[ 7], S14, 0xfd469501); /* 8 */
	FF(a, b, c, d, x[ 8], S11, 0x698098d8); /* 9 */
	FF(d, a, b, c, x[ 9], S12, 0x8b44f7af); /* 10 */
	FF(c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
	FF(b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
	FF(a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
	FF(d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
	FF(c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
	FF(b, c, d, a, x[15], S14, 0x49b40821); /* 16 */

	/* Round 2 */
	GG(a, b, c, d, x[ 1], S21, 0xf61e2562); /* 17 */
	GG(d, a, b, c, x[ 6], S22, 0xc040b340); /* 18 */
	GG(c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
	GG(b, c, d, a, x[ 0], S24, 0xe9b6c7aa); /* 20 */
	GG(a, b, c, d, x[ 5], S21, 0xd62f105d); /* 21 */
	GG(d, a, b, c, x[10], S22, 0x2441453); /* 22 */
	GG(c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
	GG(b, c, d, a, x[ 4], S24, 0xe7d3fbc8); /* 24 */
	GG(a, b, c, d, x[ 9], S21, 0x21e1cde6); /* 25 */
	GG(d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
	GG(c, d, a, b, x[ 3], S23, 0xf4d50d87); /* 27 */
	GG(b, c, d, a, x[ 8], S24, 0x455a14ed); /* 28 */
	GG(a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
	GG(d, a, b, c, x[ 2], S22, 0xfcefa3f8); /* 30 */
	GG(c, d, a, b, x[ 7], S23, 0x676f02d9); /* 31 */
	GG(b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */

	/* Round 3 */
	HH(a, b, c, d, x[ 5], S31, 0xfffa3942); /* 33 */
	HH(d, a, b, c, x[ 8], S32, 0x8771f681); /* 34 */
	HH(c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
	HH(b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
	HH(a, b, c, d, x[ 1], S31, 0xa4beea44); /* 37 */
	HH(d, a, b, c, x[ 4], S32, 0x4bdecfa9); /* 38 */
	HH(c, d, a, b, x[ 7], S33, 0xf6bb4b60); /* 39 */
	HH(b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
	HH(a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
	HH(d, a, b, c, x[ 0], S32, 0xeaa127fa); /* 42 */
	HH(c, d, a, b, x[ 3], S33, 0xd4ef3085); /* 43 */
	HH(b, c, d, a, x[ 6], S34, 0x4881d05); /* 44 */
	HH(a, b, c, d, x[ 9], S31, 0xd9d4d039); /* 45 */
	HH(d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
	HH(c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
	HH(b, c, d, a, x[ 2], S34, 0xc4ac5665); /* 48 */

	/* Round 4 */
	II(a, b, c, d, x[ 0], S41, 0xf4292244); /* 49 */
	II(d, a, b, c, x[ 7], S42, 0x432aff97); /* 50 */
	II(c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
	II(b, c, d, a, x[ 5], S44, 0xfc93a039); /* 52 */
	II(a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
	II(d, a, b, c, x[ 3], S42, 0x8f0ccc92); /* 54 */
	II(c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
	II(b, c, d, a, x[ 1], S44, 0x85845dd1); /* 56 */
	II(a, b, c, d, x[ 8], S41, 0x6fa87e4f); /* 57 */
	II(d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
	II(c, d, a, b, x[ 6], S43, 0xa3014314); /* 59 */
	II(b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
	II(a, b, c, d, x[ 4], S41, 0xf7537e82); /* 61 */
	II(d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
	II(c, d, a, b, x[ 2], S43, 0x2ad7d2bb); /* 63 */
	II(b, c, d, a, x[ 9], S44, 0xeb86d391); /* 64 */

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;

	// Zeroize sensitive information.
	// memset(x, 0, sizeof x);
}

// Encodes input (unsigned long) into output (unsigned char). Assumes len is a multiple of 4.
__device__ void update(const unsigned char input[], size_t length, byte *buffer, dword *count, dword *state)
{
	// compute number of bytes mod 64
	size_t index = bm64(count[0]);

	// Update number of bits
	if ((count[0] += (length << 3)) < (length << 3))
		count[1]++;
	count[1] += (length >> 29);

	// number of bytes we need to fill in buffer
	size_t i, firstpart = block_sz - index;

	// transform as many times as possible.
	if (length >= firstpart)
	{
		// fill buffer first, transform
		memcpy(&buffer[index], input, firstpart);
		transform(buffer, state);

		// transform chunks of block_sz (64 bytes)
		for (i = firstpart; i + block_sz <= length; i += block_sz)
			transform(&input[i], state);

		index = 0;
	}
	else i = 0;

	// buffer remaining input
	memcpy(&buffer[index], &input[i], length - i);
}

__device__ char temp[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ unsigned int length = 1;

__device__ bool MD5(char *memory, unsigned int *call)
{
	// Convert number to alpha numeric string -- 0-9a-zA-Z (62 chars)
	// Can we parallelize this? Or reduce the redudant calculations between the threads in a block?
	
	const unsigned tid = threadIdx.x;
	unsigned length;
	char buf[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	{
		qword j = ((qword) blockIdx.x) * ((qword) blockDim.x) + ((qword) tid) + 1 + ((qword) *call) * ((qword) gridDim.x) * ((qword) blockDim.x);
		qword copy = j;
		char *p = buf;
		do {
			*p++ = memory[ word + ((j-1) % charset_sz) ]; // String appears in reverse order
			j = (j-1) / charset_sz;
		} while (j > 0);
		length = p - buf;
	}

	byte buffer[block_sz];
	dword count[2] = { 0, 0 }; // 64bit counter for number of bits
	dword state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 }; // running hash

	update((const unsigned char *)buf, length, buffer, count, state);

	byte hash[word];
	// Finalize
	{
		static unsigned char padding[block_sz] = {
			0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
		};
		{
			// Save number of bits
			unsigned char bits[8];
			encode(bits, count, 8);

			// pad out to 56 mod 64.
			size_t index = bm64(count[0]);
			size_t padLen = (index < 56) ? (56 - index) : (120 - index);
			update(padding, padLen, buffer, count, state);

			// Append length (before padding)
			update(bits, 8, buffer, count, state);

			// Store state in hash
			encode(hash, state, 16);
		}
	}

	++*call;

	// Compare hash with target
	for (int i = 0; i < word; i++)
		if ((unsigned char)hash[i] != ((unsigned char)memory[i])) 
			return false;
	
	// Copy hash to memory if matched
	for (int i = 0; i < 16; i++)
		memory[i] = buf[i];
	
	return true;
}

#endif