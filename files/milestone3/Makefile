# Makefile — Milestone 3: Integration, Benchmarking, and Analysis

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

CC     := gcc
CFLAGS := -std=gnu11 -Wall -Wextra \
           -I$(INC_DIR) -I$(LLAMA_INC) -I$(LLAMA_INC2) \
           -pthread -O2 \
           -funroll-loops

LDFLAGS := -L$(LLAMA_LIB) -lllama -pthread -lm
RPATH   := -Wl,-rpath,$(realpath $(LLAMA_LIB))

MODEL      ?= ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
MAX_TOKENS ?= 128

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(RPATH)
	@echo ""
	@echo "  Build successful: ./$(TARGET)"
	@echo "  Run: ./$(TARGET) <model.gguf> [max_tokens]"
	@echo ""

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

run: all
	LD_LIBRARY_PATH=$(LLAMA_LIB):$$LD_LIBRARY_PATH \
	    ./$(TARGET) $(MODEL) $(MAX_TOKENS)

clean:
	rm -rf $(BUILD_DIR) $(TARGET) inference_engine.log
	@echo "  Cleaned."
