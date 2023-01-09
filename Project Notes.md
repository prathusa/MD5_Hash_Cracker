# Final Project Notes

This is a note sheet of my thought process while approaching this project.

Immediately, I am thinking that the hardest part will probably be designing a system that can maximize thread usage and prevent and overlap calculations.

Need to maximize thread, block, and warp usage.

Also need to figure out the project method for the user to enter a password to encrypt and then crack.

Also need to determine constraints, is the password alpha numerical with both cases? (important for the algorithm) How long can the password be (less important for the logic of the brute force system) 

I think the actual password brute-forcing logic will be simple. I think that the MD5 hash implementation will be annoying.

TODO:

- [ ]  Locking mechanism
- [ ]  Find CUDA/C++ MD5 hash library
- [ ]  Cracking algorithm

Algorithm Logic:

We want to prevent more than one thread from trying the same passcode. At the same time, we don’t want to share data between the threads since that adds additional overhead cost to each password attempt (which will be in the millions, billions, etc.). We know that each thread knows its thread id with only device-level communication (no talking to other threads). This means that space out the password attempts by the thread differential. What this means is that if we consider all possible password attempts as numbered, e.g. 0 → a, 1 → b, …, 26 → aa, …, 52 → ba, …

We also need a method to cease operation once the correct password is found.

Algorithm Psuedo Code:

```python
valid_characters # List of all possible valid characters a password can contain
while 1:
	for c in valid_characters:
		pass
# Actually I think there is a better way
'''

'''
```

Algorithm Big O (Worst Case) Analysis:

The worst case is when no password guess is correct. This results in $O (N^M)$ where **N** is the number of valid unique characters in a password and **M** is the length of the target password.

We are supposing that the conversion from string to MD5 hash is fairly constant between any two given strings.

December 10, 2022 I realized with further research that a couple of my initial ideas and suppositions were wrong.

I think I know generally what is needed for this project now, I just need to implement it.

```cpp
// Test code in t.cpp
```

December 11, 2022 I finally finished the md5 class with the help of [https://www.rfc-editor.org/rfc/rfc1321](https://www.rfc-editor.org/rfc/rfc1321), which converts strings into md5 hashs. Now I need some way to do `thread_id → alpha_num_string → hash` .

December 12, 2022 

The maximum size of an unsigned int is (unsigned int)(-1). This was causing an issue when converting from number to string since overflow was occurring which resulted in the algorithm stopping in the middle of 5 characters which is the MAX_UINT32. Fixed this by using qword which is 64 bits unsigned. No more overflow so the algorithm chugs past 5 digits.

December 14, 2022 Going to clean up the code base and start the official write-up. I played around with the threads dims and found 128 x 72 to be the best performing. Above that, the algorithm doesn’t work (which I would debug given more time). Below that performance is around 2x slower.