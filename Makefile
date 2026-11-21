CC ?= gcc
CFLAGS ?= -O3 -fPIC
CPPFLAGS ?= $(shell pkg-config --cflags vulkan 2>/dev/null)
LDFLAGS ?= $(shell pkg-config --libs vulkan 2>/dev/null || echo -lvulkan)
PYTHON ?= python3

.PHONY: all shaders clean

all: libvk_inkling.so libdump_spv.so

libvk_inkling.so: vk_inkling.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -shared $< -o $@ $(LDFLAGS)

libdump_spv.so: dump_spv.c
	$(CC) $(CPPFLAGS) -O2 -fPIC -shared $< -o $@

shaders: libdump_spv.so VK_LAYER_DUMP_SPV.json
	mkdir -p spv_batch spv_prefill
	VK_LAYER_PATH=$(CURDIR) VK_INSTANCE_LAYERS=VK_LAYER_DUMP_SPV DUMP_SPV_DIR=$(CURDIR)/spv_batch $(PYTHON) build_batched_spv.py
	VK_LAYER_PATH=$(CURDIR) VK_INSTANCE_LAYERS=VK_LAYER_DUMP_SPV DUMP_SPV_DIR=$(CURDIR)/spv_prefill $(PYTHON) build_prefill_spv.py

clean:
	rm -f libvk_inkling.so libdump_spv.so llama_argmax500
	rm -rf spv_batch spv_prefill __pycache__
