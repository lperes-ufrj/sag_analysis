CXX      := clang++
CXXFLAGS := -O2 -Wall -std=c++17 $(shell root-config --cflags)
LDLIBS   := $(shell root-config --libs)

TARGET := read_waveforms        # nome do executável de saída
SRC    := read_waveforms.cpp    # seu arquivo-fonte

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)