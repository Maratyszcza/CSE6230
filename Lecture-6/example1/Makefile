all:
	$(CXX) -O2 -msse2 -I. -c -o compute.o compute.cpp
	$(CXX) -O2 -msse2 -I. -c -o main.o main.cpp
	$(CXX) -o main main.o compute.o

clean:
	rm *.o
	rm main
