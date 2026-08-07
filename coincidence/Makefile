CXX ?= c++
CXXFLAGS := -O3 -Wall -Wextra -pedantic -std=c++17 $(shell root-config --cflags)
LDLIBS := $(shell root-config --libs)

TARGET := run_coincidence
SOURCES := run_coincidence.cpp coincidence_lib.cpp
HEADERS := coincidence_lib.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@ $(LDLIBS)

clean:
	$(RM) $(TARGET)
