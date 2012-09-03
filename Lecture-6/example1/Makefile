all:
	$(CXX) $(CXXFLAGS) -I. -c -o compute.o compute.cpp
	$(CXX) $(CXXFLAGS) -I. -c -o main.o main.cpp
	$(CXX) -o main main.o compute.o

clean:
	rm *.o
	rm main
