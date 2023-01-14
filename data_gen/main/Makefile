main:main.cpp
	g++ -O3 -mtune=native -march=native -mfpmath=both -o main -I /usr/include/eigen3 -I ../../qulacs/include -I ../include -L ../../qulacs/lib main.cpp ../src/DataCreator.cpp -fopenmp -lcppsim_static -lcsim_static

icpc:main.cpp
	icpc -fast -o main_icpc -I ../../qulacs/include -L ../../qulacs/lib -I ../include main.cpp ../src/DataCreator.cpp -qopenmp -lcppsim_static -lcsim_static

icpc_parallel:main.cpp
	icpc -fast -o main_icpc_parallel -I ../../qulacs/include -I ../include -L ../../qulacs/lib main.cpp ../src/DataCreator.cpp -qopenmp -parallel  -lcppsim_static -lcsim_static

clean:
	rm -f main
	rm -f main_icpc
	rm -f main_icpc_parallel

clean_data:
	rm -f ../result/*txt
	rm -f ../result/*csv
	rm -f *.o*
	rm -f *.e*

clean_all:
	rm -f main
	rm -f main_icpc
	rm -f main_icpc_parallel
	rm -f ../result/*.txt
	rm -f ../result/*.csv
	rm -f *.o*
	rm -f *.e*
