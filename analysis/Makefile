CXX ?= c++
CXXFLAGS := -O3 -Wall -Wextra -pedantic -std=c++17 $(shell root-config --cflags)
CPPFLAGS := -DANALYSIS_DIR='"$(abspath .)"'
LDLIBS := $(shell root-config --libs)

BIN_DIR := ../bin
TARGET := $(BIN_DIR)/plot_allwaveforms

.PHONY: all clean plot_allwaveforms

all: $(TARGET)

plot_allwaveforms: $(TARGET)

$(BIN_DIR):
	mkdir -p $@

$(TARGET): plot_allwaveforms.cpp | $(BIN_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

clean:
	$(RM) $(TARGET)
