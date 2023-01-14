main:calc_fp.cpp
	g++ -O3 -mtune=native -march=native -mfpmath=both -o main -I ../qulacs/include -L ../qulacs/lib calc_fp.cpp -fopenmp -lcppsim_static -lcsim_static -std=c++1z -lstdc++fs

clean:
	rm main

clean_data:
	rm ./result/*
