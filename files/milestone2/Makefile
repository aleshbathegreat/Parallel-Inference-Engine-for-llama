# ──────────────────────────────────────────────────────────────────────────────
# Makefile — Parallel Inference Engine (Milestone 2)
#
# Prerequisites
# ─────────────
#   LLAMA_DIR  = path to llama.cpp root (must be built with -DBUILD_SHARED_LIBS=ON)
#
# Usage
# ─────
#   make LLAMA_DIR=../llama.cpp
#   make run  LLAMA_DIR=../llama.cpp MODEL=./tinyllama.gguf N_WORKERS=4 MAX_TOKENS=128
#   make clean
# ──────────────────────────────────────────────────────────────────────────────

LLAMA_DIR   ?= ../llama.cpp
LLAMA_INC   := $(LLAMA_DIR)/include
LLAMA_INC2  := $(LLAMA_DIR)/ggml/include
LLAMA_LIB   := $(LLAMA_DIR)/build/bin

SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
TARGET      := inference_engine

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

CC      := gcc
CFLAGS  := -std=gnu11 -Wall -Wextra \
            -I$(INC_DIR) \
            -I$(LLAMA_INC) \
            -I$(LLAMA_INC2) \
            -pthread \
            -O2 -g

LDFLAGS := -L$(LLAMA_LIB) -lllama -pthread -lm
RPATH   := -Wl,-rpath,$(realpath $(LLAMA_LIB))

# Demo defaults
MODEL      ?= ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
N_WORKERS  ?= 2
MAX_TOKENS ?= 128

.PHONY: all clean run check-llama

all: check-llama $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(RPATH)
	@echo ""
	@echo "  Build successful: ./$(TARGET)"
	@echo "  Run: ./$(TARGET) <model.gguf> [n_workers] [max_tokens]"
	@echo ""

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

check-llama:
	@if [ ! -f "$(LLAMA_INC)/llama.h" ]; then \
	    echo "ERROR: llama.h not found at $(LLAMA_INC)"; \
	    echo "Set LLAMA_DIR to your llama.cpp root."; \
	    exit 1; \
	fi
	@if [ ! -d "$(LLAMA_LIB)" ]; then \
	    echo "WARNING: $(LLAMA_LIB) not found. Build llama.cpp first."; \
	fi

run: all
	LD_LIBRARY_PATH=$(LLAMA_LIB):$$LD_LIBRARY_PATH \
	    ./$(TARGET) $(MODEL) $(N_WORKERS) $(MAX_TOKENS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) inference_engine.log
	@echo "  Cleaned."
