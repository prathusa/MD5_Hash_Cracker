#ifndef MD5_H
#define MD5_H

#include <cstring>
#include <iostream>
#include <string>

class MD5
{
	typedef unsigned char byte;
	typedef unsigned int dword;
	const static unsigned int block_sz = 64;

public:
	const static unsigned int hash_sz = 128;
	const static std::size_t size = hash_sz / (sizeof(byte) * 8); // length of the hash in bytes
	byte hash[size]; // the result

	MD5(const unsigned char* buf, std::size_t length)
	{
		update(buf, length);
		finalize();
	}
	MD5(const std::string& text) : MD5((const unsigned char*)text.c_str(), text.length()) {}
	MD5& finalize();
	operator std::string() const; // repr / to_string
	operator char*() const { return (char *)hash; }; // cast to char*, can use for faster comparison with strncmp((char *)res, size)
	bool operator == (const char* other) const { return memcmp((char *)hash, other, hash_sz) == 0; }; // compare with char*
	static inline std::string md5(const std::string str) { return MD5(str); };
	static inline char *md5(const char *str) { return MD5(str); };
	static void byte_hash(const char *str_hash, char *byte_hash);
	static void str_hash(const char *byte_hash, char *str_hash);

private:
	bool finalized = false;
	
	byte buffer[block_sz]; // excess bytes that don't fit into block_sz
	dword count[2] = { 0, 0 }; // 64bit counter for number of bits (lo, hi)
	dword state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 }; // running hash

	// inline operations
	static inline dword bm64(dword x) { return x >> 3 & 0x7f; }; // dword bytes mod 64
	static inline dword F(dword x, dword y, dword z) { return (x & y) | (~x & z); };
	static inline dword G(dword x, dword y, dword z) { return (x & z) | (y & ~z); };
	static inline dword H(dword x, dword y, dword z) { return x ^ y ^ z; };
	static inline dword I(dword x, dword y, dword z) { return y ^ (x | ~z); }
	static inline dword rotate_left(dword x, int n) { return (x << n) | (x >> (32 - n)); };
	static inline void FF(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + F(b, c, d) + x + ac, s) + b; };
	static inline void GG(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + G(b, c, d) + x + ac, s) + b; };
	static inline void HH(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + H(b, c, d) + x + ac, s) + b; };
	static inline void II(dword& a, dword b, dword c, dword d, dword x, dword s, dword ac) { a = rotate_left(a + I(b, c, d) + x + ac, s) + b; };

	static void encode(byte output[], const dword input[], std::size_t len);
	static void decode(dword output[], const byte input[], std::size_t len);
	void transform(const byte block[block_sz]);
	void update(const unsigned char* buf, std::size_t length);

};




#endif