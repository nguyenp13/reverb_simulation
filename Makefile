CXX=g++
CXXFLAGS=-fmax-errors=3

all: CXXFLAGS += -O3 -funroll-loops 
all: LDFLAGS += -lsndfile

all: main
	
main: main.cpp
	$(CXX) -o main $(CXXFLAGS) main.cpp $(LDFLAGS)

clean:
	rm -f main
