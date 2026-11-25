CXX      := g++
CXXFLAGS := -O3 -fopenmp -std=c++17
LDFLAGS  := -lvulkan -lpthread
GLSLC    := glslc
GLSLFLAGS:= --target-env=vulkan1.3 -O

SHADERS  := $(wildcard shaders/*.comp)
SPVS     := $(SHADERS:.comp=.spv)

.PHONY: all shaders clean

all: shaders inkling_vk

shaders: $(SPVS)

shaders/%.spv: shaders/%.comp
	$(GLSLC) $(GLSLFLAGS) $< -o $@

inkling_vk: inkling_vk.cpp vk_abi.h
	$(CXX) $(CXXFLAGS) inkling_vk.cpp $(LDFLAGS) -o $@

vk_probe: vk_probe.cpp vk_abi.h
	$(CXX) -O3 -std=c++17 vk_probe.cpp $(LDFLAGS) -o $@

ref_cpu: ref_cpu.cpp
	$(CXX) -O3 -fopenmp -std=c++17 ref_cpu.cpp -o $@

clean:
	rm -f inkling_vk vk_probe ref_cpu bench_ops shaders/*.spv
