compile: main

PHONY: clean t

te: t.cu
	nvcc t.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o t
	./t

# test: main.cu cracker_test.cu cracker.cuh
# 	nvcc main.cu cracker_test.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o main
# 	sbatch run.sh
# 	sleep 5
# 	cat main.out

main: main.cu cracker.cu cracker.cuh md5.h md5.cpp md5.cuh
	nvcc main.cu cracker.cu md5.cpp -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -g -std c++17 -o main -diag-suppress=177 -lineinfo

test: main
	./main cfcd208495d565ef66e7dff9f98764da # '0'
	./main 21c2e59531c8710156d34a3c30ac81d5 # 'Z'
	./main 25ed1bcb423b0b7200f485fc5ff71c8e # 'zz'
	./main e1faffb3e614e6c2fba74296962386b7 # 'AAA'
	./main 2524fb65ffd8bb8378633440f850db73 # 'LIP' best owl flex hitscan
	./main 6057f13c496ecf7fd777ceb9e79ae285 # 'hey'
	./main 23a58bf9274bedb19375e527a0744fa9 # 'with'
	./main aeb0b865236c3d5bdabecd8b21002063 # 'rizz' ;)
	./main 5d41402abc4b2a76b9719d911017c592 # 'hello'
	./main 7d793037a0760186574b0282f2f435e7 # 'world' classic string
	./main 9c06717b82bb9d40e33b8274a4dc032e # 'ece759' fun class
	./main 29c3eea3f305d6b823f562ac4be35217 # '0000000' first permutation of 7 letter string
	./main 7fa8282ad93047a4d6fe6111c93b308a # '1111111' 
	./main d3fef2d32b68163b18ae0db519ed4de1 # 'ALLCAPS'
	./main 8467439668f26d5318f4152e0a0aecae # 'fromis9' elite group
	./main 916172e7995de7e1e64c0c3aad1edd59 # 'pratham' my name
	./main f0e8fb430bbdde6ae9c879a518fd895f # 'zzzzzzz' last permutation of 7 letter string

t: t.cpp 
	g++ -O3 -g3 -std=c++17 t.cpp -o t
	./t

test_euler: main
	sbatch test.sh

hash: md5.cpp md5.h test_hash.cpp
	g++ -O3 -Wall -g3 -std=c++17 md5.cpp test_hash.cpp -o hash
	./hash

clean:
	rm -f main
	rm -f main.out
	rm -f t
	rm -f hash

