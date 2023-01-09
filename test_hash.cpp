#include "md5.h"

#include <iostream>
#include <cassert>

using std::cout;

void test(const std::string& input, const std::string& expected)
{
    cout << "Testing \'" << input << "\' ..." << '\n';
    assert (MD5::md5(input) == expected);
    // cout << ((char *)md5(input.c_str())) << '\n';
    cout << "Passed!" << '\n';
}

int main(int argc, char *argv[])
{
    test("hash", "0800fc577294c34e0b28ad2839435945");
    test("420", "b6f0479ae87d244975439c6124592772");
    test("pratham", "916172e7995de7e1e64c0c3aad1edd59");
    test("ece759", "9c06717b82bb9d40e33b8274a4dc032e");

    cout << "Byte Hash Test ..." << '\n';
    char byte_hash[16];
    MD5::byte_hash("0800fc577294c34e0b28ad2839435945", byte_hash);
    char byte_hash_str[33];
    // for (int i = 0; i < 16; i++)
	// 	sprintf(&byte_hash_str[2*i], "%02hhx", byte_hash[i]);
    byte_hash_str[32] = 0;
    cout << byte_hash_str << '\n';
    MD5 hash("hash");
    // cout << memcmp(hash.hash, byte_hash, 16) << '\n';
    assert (memcmp(hash.hash, byte_hash, 16) == 0);
    cout << "Passed!" << '\n';
    cout << "All tests passed!" << std::endl;
    return 0;
}