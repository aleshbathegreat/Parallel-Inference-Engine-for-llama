# ──────────────────────────────────────────────────────────────────────────────
# Makefile — Parallel Inference Engine (Milestone 1)
#
# Prerequisites
# ─────────────
#  1. Clone llama.cpp:
#       git clone https://github.com/ggerganov/llama.cpp.git
#       cd llama.cpp && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build
#
#  2. Set LLAMA_DIR to the root of your llama.cpp clone:
#       make LLAMA_DIR=/path/to/llama.cpp
#     OR export it in your shell:
#       export LLAMA_DIR=/path/to/llama.cpp
#
#  3. Download a TinyLLaMA GGUF model, e.g.:
#       huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
#         tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir .
#
# Targets
# ───────
#   make              — Build the inference engine binary
#   make clean        — Remove build artefacts
#   make run          — Build and run with default arguments
#   make run-verbose  — Build and run with DEBUG-level logging
# ──────────────────────────────────────────────────────────────────────────────

# ── Paths ─────────────────────────────────────────────────────────────────────
LLAMA_DIR   ?= ../llama.cpp
LLAMA_INC   := $(LLAMA_DIR)/include
LLAMA_LIB   := $(LLAMA_DIR)/build/bin    # libllama.so lives here after cmake build

SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build

TARGET      := inference_engine

# ── Sources ───────────────────────────────────────────────────────────────────
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# ── Compiler flags ────────────────────────────────────────────────────────────
CC      := gcc
CFLAGS  := -std=gnu11 -Wall -Wextra \
            -I$(INC_DIR) \
            -I$(LLAMA_INC) \
	    -I$(LLAMA_DIR)/ggml/include \
            -pthread \
            -O2 -g

# Uncomment to enable AddressSanitizer during development:
# CFLAGS += -fsanitize=address,undefined

# ── Linker flags ──────────────────────────────────────────────────────────────
LDFLAGS := -L$(LLAMA_LIB) \
            -lllama \
            -pthread \
            -lm

# Tell the dynamic linker where to find libllama at runtime.
# On macOS use DYLD_LIBRARY_PATH instead.
RPATH := -Wl,-rpath,$(realpath $(LLAMA_LIB))

# ── Demo defaults (override on command line) ──────────────────────────────────
MODEL       ?= ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
N_THREADS   ?= 4
MAX_TOKENS  ?= 64

# ──────────────────────────────────────────────────────────────────────────────
# Rules
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: all clean run run-verbose check-llama

all: check-llama $(TARGET)

## Link
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(RPATH)
	@echo ""
	@echo "  Build successful: ./$(TARGET)"
	@echo "  Usage: ./$(TARGET) <model.gguf> [n_threads] [max_tokens]"
	@echo ""

## Compile each .c → .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

## Sanity-check that LLAMA_DIR looks correct
check-llama:
	@if [ ! -f "$(LLAMA_INC)/llama.h" ]; then \
	    echo ""; \
	    echo "  ERROR: cannot find llama.h at $(LLAMA_INC)/llama.h"; \
	    echo "  Please set LLAMA_DIR to the root of your llama.cpp clone."; \
	    echo "  Example:  make LLAMA_DIR=/path/to/llama.cpp"; \
	    echo ""; \
	    exit 1; \
	fi
	@if [ ! -d "$(LLAMA_LIB)" ]; then \
	    echo ""; \
	    echo "  WARNING: $(LLAMA_LIB) does not exist."; \
	    echo "  Build llama.cpp first:"; \
	    echo "    cd $(LLAMA_DIR) && cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build"; \
	    echo ""; \
	fi

## Run with default settings
run: all
	@if [ ! -f "$(MODEL)" ]; then \
	    echo "  ERROR: model file '$(MODEL)' not found."; \
	    echo "  Set MODEL=/path/to/your/model.gguf"; \
	    exit 1; \
	fi
	LD_LIBRARY_PATH=$(LLAMA_LIB):$$LD_LIBRARY_PATH \
	    ./$(TARGET) $(MODEL) $(N_THREADS) $(MAX_TOKENS)

## Run with full DEBUG output
run-verbose: all
	@if [ ! -f "$(MODEL)" ]; then \
	    echo "  ERROR: model file '$(MODEL)' not found."; \
	    exit 1; \
	fi
	LD_LIBRARY_PATH=$(LLAMA_LIB):$$LD_LIBRARY_PATH \
	    ./$(TARGET) $(MODEL) $(N_THREADS) $(MAX_TOKENS) 2>&1 | tee run_verbose.log

## Remove build artefacts
clean:
	rm -rf $(BUILD_DIR) $(TARGET) inference_engine.log run_verbose.log
	@echo "  Cleaned."
