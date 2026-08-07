CXX ?= c++
CXXFLAGS := -O3 -Wall -Wextra -pedantic -std=c++17 $(shell root-config --cflags)
LDLIBS := $(shell root-config --libs)

TARGETS := run_coincidence plot_wfs_coincidence
HEADERS := coincidence_lib.hpp

.PHONY: all clean

all: $(TARGETS)

run_coincidence: run_coincidence.cpp coincidence_lib.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) run_coincidence.cpp coincidence_lib.cpp -o $@ $(LDLIBS)

plot_wfs_coincidence: plot_wfs_coincidence.cpp coincidence_lib.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) plot_wfs_coincidence.cpp coincidence_lib.cpp -o $@ $(LDLIBS)

clean:
	$(RM) $(TARGETS)
