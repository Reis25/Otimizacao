# Compiler binary:
CPP=g++

# Recommended compiler flags for speed:
#	OpenMP enabled
#	full binary code optimization
#	full error and warning reports
#	no range checking within BRKGA:
CFLAGS=-O3  -Wextra -Wall -Weffc++ -ansi -pedantic -Woverloaded-virtual -Wcast-align -Wpointer-arith

# Compiler flags for debugging; uncomment if needed:
#	range checking enabled in the BRKGA API
#	OpenMP disabled
#	no binary code optimization
#CFLAGS= -DRANGECHECK -Wextra -Wall -Weffc++ -ansi -pedantic -Woverloaded-virtual -Wcast-align -Wpointer-arith

# Objects:
OBJECTS= Population.o LevitacaoDecoder.o api-levitacao.o

# Targets:
all:api-levitacao

api-levitacao: $(OBJECTS)
	$(CPP) $(CFLAGS) $(OBJECTS) -o api-levitacao

api-levitacao.o:
	$(CPP) $(CFLAGS) -c api-levitacao.cpp

LevitacaoDecoder.o:
	$(CPP) $(CFLAGS) -c LevitacaoDecoder.cpp

Population.o:
	$(CPP) $(CFLAGS) -c brkgaAPI/Population.cpp

clean:
	rm -f *.o api-levitacao
