#include "cracker.cuh"
#include "md5.h"

#include <iostream>
#include <string>
#include <stdio.h>
#include <memory>

using namespace std;


int main( int argc, char **argv )
{
    char byte_hash[MD5::size+1];
    byte_hash[MD5::size] = '\0';
    
    if (argc > 2)
    {
        printf("Usage: %s <hash>", argv[0]);
        return 1;
    }
    else if (argc == 1)
    {
        string buf;
        cout << "Please enter a string to time: ";
        cin >> buf;
        MD5::byte_hash(MD5::md5(buf).c_str(), byte_hash);
        // Print hash
        cout << "Attempting to crack hash: ";
        for (int i = 0; i < (int) MD5::size; i++)
            printf("%02hhx", byte_hash[i]);
        cout << '\n';
    }
    else
    {
        MD5::byte_hash(argv[1], byte_hash);
    }

    // Result
    char code[MD5::size];
    float duration = crack(byte_hash, code);

    cout << "Solved! \'" << code << "\' in " << duration << " ms." << '\n';

    return 0;
}