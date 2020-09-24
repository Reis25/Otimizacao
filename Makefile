CFLAGS=-O3 -fopenmp -static-libstdc++ -static-libgcc -Wall

all:main.o SampleDecoder.o
	g++  main.o SampleDecoder.o -o main.run $(CPLEXFLAGS) $(CFLAGS)

%.o: %.cpp %.hpp 
	g++ -c $< -o $@ $(CFLAGS)

main.o: main.cpp
	g++ -c -o main.o main.cpp $(CPLEXFLAGS) $(CFLAGS)

clean:
	rm -f *.o
