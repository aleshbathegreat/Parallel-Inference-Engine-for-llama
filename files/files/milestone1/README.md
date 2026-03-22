# Parallel Inference Engine — Milestone 1
### Core Inference Orchestration Layer

---

## Overview

This is a complete C implementation of **Milestone 1** of the Parallel LLM Inference Engine project. It implements a correct, thread-safe inference orchestration layer on top of TinyLLaMA via [llama.cpp](https://github.com/ggerganov/llama.cpp).

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Producer Threads (N)                    │
│    engine_submit_sync()    engine_submit_async()         │
└────────────────────┬────────────────────────────────────┘
                     │  InferenceJob*  (heap-allocated)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   JobQueue  (thread-safe FIFO)           │
│   mutex + cond_not_empty + cond_not_full                 │
│   Blocks producers if full; blocks worker when empty     │
└────────────────────┬────────────────────────────────────┘
                     │  queue_pop()  (blocking)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Worker Thread (1)                       │
│   job_mark_running() → run_inference() → job_mark_done() │
│                                                          │
│   run_inference():                                       │
│     1. llama_tokenize()      — prompt → token IDs        │
│     2. llama_decode(batch)   — prefill KV cache          │
│     3. Loop: sample token → detokenise → append output   │
│     4. Stop on EOG or max_tokens                         │
└─────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **Single worker thread** | `llama_context` is not re-entrant. Milestone 2 will add multiple contexts for parallel execution. |
| **Linked-list queue** | Unbounded in memory, O(1) push/pop. A fixed-size ring buffer could be used in Milestone 2. |
| **Per-job mutex + cond** | Allows `job_wait()` to block efficiently without polling. |
| **KV cache cleared between jobs** | Simple and correct for Milestone 1. Milestone 2 can explore context reuse. |
| **Greedy + top-p sampler in C** | No dependency on llama.cpp's sampler API, which changes between versions. |

---

## File Structure

```
milestone1/
├── include/
│   ├── logger.h     — Thread-safe logging (DEBUG/INFO/WARN/ERROR)
│   ├── job.h        — InferenceJob struct and lifecycle API
│   ├── queue.h      — Thread-safe FIFO job queue
│   └── engine.h     — InferenceEngine orchestration layer
├── src/
│   ├── logger.c     — Logger implementation
│   ├── job.c        — Job lifecycle (create, mark_running/done/error, wait)
│   ├── queue.c      — Queue push/pop with mutex + condition variables
│   ├── engine.c     — Model loading, worker thread, inference loop
│   └── main.c       — Demo: sequential correctness + concurrent stress test
├── Makefile
└── README.md
```

---

## Setup

### Step 1 — Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=ON
cmake --build build --config Release -j$(nproc)
cd ..
```

### Step 2 — Download TinyLLaMA GGUF model

```bash
# Option A: via huggingface-cli
pip install huggingface_hub
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
    tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    --local-dir ./milestone1

# Option B: direct wget (check HuggingFace for the current URL)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -P ./milestone1
```

### Step 3 — Build the project

```bash
cd milestone1
make LLAMA_DIR=../llama.cpp
```

---

## Running

```bash
# Default: 4 CPU threads, 64 max tokens per request
./inference_engine ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Custom thread count and token limit
./inference_engine ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 8 128

# Or use the make target (MODEL, N_THREADS, MAX_TOKENS are configurable)
make run LLAMA_DIR=../llama.cpp MODEL=./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Full verbose / debug output
make run-verbose LLAMA_DIR=../llama.cpp
```

If `libllama.so` is not on your default library path, prefix with:

```bash
LD_LIBRARY_PATH=../llama.cpp/build/src:$LD_LIBRARY_PATH ./inference_engine ...
```

---

## What the demo does

**Phase 1 — Sequential correctness check**

Three prompts are submitted one at a time (synchronously). This verifies that:
- The model loads and generates coherent output.
- Timing metadata is recorded correctly (queue wait, exec time, total).
- The KV cache is correctly cleared between jobs.

**Phase 2 — Concurrent submission stress test**

`N_PRODUCER_THREADS` (4) threads each submit `JOBS_PER_THREAD` (3) requests simultaneously, alternating between `engine_submit_sync()` and `engine_submit_async()`. This verifies:
- The job queue correctly serialises concurrent submissions.
- `job_wait()` wakes up the right thread after each job completes.
- No output corruption occurs (each job's output buffer is private).
- Stats counters are accurate under concurrent load.

---

## Sample output

```
Loading model, please wait...
Engine running.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 1: Sequential correctness check
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Sequential 1/3]
  Prompt : The capital of Japan is
  Status : DONE
  Output :  Tokyo, the largest city in Japan.
  Timing : queue=0.1 ms  exec=1823.4 ms  total=1823.5 ms

...

╔══════════════════════════════════════╗
║        Engine Statistics             ║
╠══════════════════════════════════════╣
║  Jobs submitted  : 15                ║
║  Jobs completed  : 15                ║
║  Jobs failed     : 0                 ║
║  Avg exec time   :        1910.3 ms  ║
║  Avg queue wait  :          12.5 ms  ║
║  Total exec time :       28654.8 ms  ║
╚══════════════════════════════════════╝
```

---

## Milestone 2 extension points

The codebase is deliberately structured to make Milestone 2 straightforward:

| Component | Milestone 2 change |
|---|---|
| `engine.c` | Add `n_workers` contexts; spawn N worker threads |
| `queue.c` | Already handles N consumers correctly |
| `engine.c:run_inference` | Add batch prefill across multiple jobs |
| `main.c` | Add throughput benchmarking with higher load |

---

## Troubleshooting

**`cannot find llama.h`** — Set `LLAMA_DIR` to the llama.cpp repo root.

**`libllama.so: not found` at runtime** — Use `LD_LIBRARY_PATH` as shown above, or install the library system-wide.

**Segfault / assertion in llama.cpp** — The context window (`n_ctx`) may be too small for the prompt + max_tokens. Reduce max_tokens or increase `n_ctx` in the config.

**Garbled output** — Ensure the GGUF file is not truncated (check the download).
