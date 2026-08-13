CXX ?= c++
CXXFLAGS := -O3 -Wall -Wextra -pedantic -std=c++17 $(shell root-config --cflags)
LDLIBS := $(shell root-config --libs)

BIN_DIR := ../bin
PROGRAMS := run_coincidence plot_wfs_coincidence
TARGETS := $(addprefix $(BIN_DIR)/,$(PROGRAMS))
HEADERS := coincidence_lib.hpp

.PHONY: all clean $(PROGRAMS)

all: $(TARGETS)

run_coincidence: $(BIN_DIR)/run_coincidence

plot_wfs_coincidence: $(BIN_DIR)/plot_wfs_coincidence

$(BIN_DIR):
	mkdir -p $@

$(BIN_DIR)/run_coincidence: run_coincidence.cpp coincidence_lib.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) run_coincidence.cpp coincidence_lib.cpp -o $@ $(LDLIBS)

$(BIN_DIR)/plot_wfs_coincidence: plot_wfs_coincidence.cpp coincidence_lib.cpp $(HEADERS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) plot_wfs_coincidence.cpp coincidence_lib.cpp -o $@ $(LDLIBS)

clean:
	$(RM) $(TARGETS)
