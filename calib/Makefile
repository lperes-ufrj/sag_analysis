CXX      := clang++
CXXFLAGS := -O2 -Wall -std=c++17 $(shell root-config --cflags)
LDLIBS   := $(shell root-config --libs)

TARGET := read_waveforms
# Add the utility cpp file here!
SRC    := read_waveforms.cpp aux/waveform_utils.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDLIBS)

clean:
	rm -f $(TARGET)