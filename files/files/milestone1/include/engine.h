#ifndef ENGINE_H
#define ENGINE_H

/*
 * engine.h — Core inference orchestration layer (Milestone 1)
 *
 * The InferenceEngine owns:
 *   - The loaded TinyLLaMA model and its inference context
 *   - A thread-safe job queue
 *   - A single worker thread that drains the queue sequentially
 *
 * Design rationale (Milestone 1)
 * ────────────────────────────────
 * Milestone 1 focuses on *correctness* and *system structure*.  A single
 * worker serialises all llama.cpp calls (the context is not re-entrant),
 * while the queue decouples producers from the worker.  Milestone 2 will
 * introduce parallel workers, batching, and resource limits.
 *
 * Usage pattern
 * ─────────────
 *   EngineConfig cfg = ENGINE_CONFIG_DEFAULT;
 *   snprintf(cfg.model_path, sizeof(cfg.model_path), "/path/to/model.gguf");
 *
 *   InferenceEngine *eng = engine_create(&cfg);
 *   engine_start(eng);
 *
 *   // Synchronous call — blocks until the result is ready
 *   InferenceJob *job = engine_submit_sync(eng, "Hello, world!", 128, 0.0f);
 *   printf("Output: %s\n", job->output);
 *   job_destroy(job);
 *
 *   engine_stop(eng);
 *   engine_destroy(eng);
 */

#include <stdint.h>
#include <pthread.h>
#include "job.h"
#include "queue.h"

/* Forward-declare llama types so callers do not need llama.h */
struct llama_model;
struct llama_context;

/* ── Configuration ──────────────────────────────────────────────────────── */
typedef struct {
    char  model_path[512];      /* Path to the .gguf model file          */
    int   n_ctx;                /* KV-cache / context window size        */
    int   n_threads;            /* CPU threads for inference             */
    int   n_gpu_layers;         /* Layers offloaded to GPU (0 = CPU-only)*/
    int   max_queue_size;       /* Max pending jobs (0 = unlimited)      */
    int   default_max_tokens;   /* Fallback if caller passes ≤ 0         */
    float default_temperature;  /* Fallback if caller passes < 0         */
    int   seed;                 /* RNG seed for sampling (-1 = random)   */
} EngineConfig;

/* Sensible defaults — copy this and override the model_path field. */
#define ENGINE_CONFIG_DEFAULT {         \
    .model_path          = "",          \
    .n_ctx               = 2048,        \
    .n_threads           = 4,           \
    .n_gpu_layers        = 0,           \
    .max_queue_size      = 256,         \
    .default_max_tokens  = 256,         \
    .default_temperature = 0.0f,        \
    .seed                = -1           \
}

/* ── Runtime statistics ─────────────────────────────────────────────────── */
typedef struct {
    uint64_t jobs_submitted;
    uint64_t jobs_completed;
    uint64_t jobs_failed;
    double   total_exec_ms;       /* Sum of all individual job exec times */
    double   total_queue_wait_ms; /* Sum of all queue-wait times          */
} EngineStats;

/* ── Engine ─────────────────────────────────────────────────────────────── */
typedef struct {
    EngineConfig config;

    /* llama.cpp handles */
    struct llama_model   *model;
    struct llama_context *ctx;

    /* Concurrency */
    JobQueue    *queue;
    pthread_t    worker_thread;
    volatile int running;        /* 1 = worker is active, 0 = stopped    */

    /* Atomic-ish stats (protected by stats_mutex) */
    EngineStats      stats;
    pthread_mutex_t  stats_mutex;

    /* Monotonically increasing job ID counter */
    uint64_t         next_job_id;
    pthread_mutex_t  id_mutex;
} InferenceEngine;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

/**
 * engine_create() — Allocate the engine, load the model, and prepare the
 *                   inference context.
 *
 * @param config  Pointer to a fully populated EngineConfig.
 * @return        Heap-allocated engine, or NULL on failure.
 */
InferenceEngine *engine_create(const EngineConfig *config);

/**
 * engine_start() — Spawn the worker thread and begin processing jobs.
 *
 * @return 0 on success, -1 on error.
 */
int engine_start(InferenceEngine *engine);

/**
 * engine_stop() — Drain the queue, stop the worker, and join the thread.
 *  After this call the engine is idle but still valid; engine_destroy()
 *  must be called separately to free resources.
 */
void engine_stop(InferenceEngine *engine);

/**
 * engine_destroy() — Free all resources owned by the engine.
 *  Must be called after engine_stop().
 */
void engine_destroy(InferenceEngine *engine);

/* ── Job submission ─────────────────────────────────────────────────────── */

/**
 * engine_submit_sync() — Submit a job and block until it completes.
 *
 * The returned job is heap-allocated and owned by the caller.
 * Call job_destroy() when done.
 *
 * @param prompt       Null-terminated input prompt.
 * @param max_tokens   Max tokens to generate; ≤ 0 uses the configured default.
 * @param temperature  Sampling temperature; < 0 uses the configured default.
 * @return             Completed job (status DONE or ERROR), or NULL on
 *                     submission failure.
 */
InferenceJob *engine_submit_sync(InferenceEngine *engine,
                                 const char *prompt,
                                 int max_tokens,
                                 float temperature);

/**
 * engine_submit_async() — Submit a job and return immediately.
 *
 * The job status starts as JOB_STATUS_PENDING.  Call job_wait() to block
 * until it transitions to DONE or ERROR, then call job_destroy() to free it.
 *
 * @return  Job handle, or NULL on submission failure.
 */
InferenceJob *engine_submit_async(InferenceEngine *engine,
                                  const char *prompt,
                                  int max_tokens,
                                  float temperature);

/* ── Utilities ──────────────────────────────────────────────────────────── */

/**
 * engine_get_stats() — Copy the current stats into *out.
 */
void engine_get_stats(const InferenceEngine *engine, EngineStats *out);

/**
 * engine_print_stats() — Print a human-readable stats summary to stdout.
 */
void engine_print_stats(const InferenceEngine *engine);

#endif /* ENGINE_H */
