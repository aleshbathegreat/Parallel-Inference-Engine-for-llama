#ifndef ENGINE_H
#define ENGINE_H

/*
 * engine.h — Parallel Inference Engine (Milestone 2)
 *
 * N worker threads each own a separate llama_context so that
 * llama_decode() calls never conflict.  The llama_model is shared
 * and read-only after loading.
 *
 *   ┌─ producer threads ─┐
 *   │  submit_sync/async  │
 *   └────────┬────────────┘
 *            │
 *       [ JobQueue ]          ← single shared FIFO (thread-safe)
 *            │
 *   ┌────────┼────────────────────────┐
 *   ▼        ▼                        ▼
 * Worker-0  Worker-1  ...  Worker-(N-1)
 * (ctx-0)  (ctx-1)        (ctx-N-1)
 */

#include <stdint.h>
#include <pthread.h>
#include "job.h"
#include "queue.h"

struct llama_model;
struct llama_context;

/* ── Configuration ──────────────────────────────────────────────────────── */
typedef struct {
    char  model_path[512];
    int   n_ctx;                /* KV-cache size per context              */
    int   n_threads;            /* CPU threads per worker context         */
    int   n_gpu_layers;
    int   n_workers;            /* Number of parallel worker threads      */
    int   max_queue_size;
    int   default_max_tokens;
    float default_temperature;
    int   seed;
    /* Optional: pass an already-loaded model to avoid loading it again.
     * If non-NULL, engine_create() skips loading and engine_destroy()
     * skips freeing it — the caller owns the model lifetime. */
    struct llama_model *preloaded_model;
} EngineConfig;

#define ENGINE_CONFIG_DEFAULT {         \
    .model_path          = "",          \
    .n_ctx               = 2048,        \
    .n_threads           = 4,           \
    .n_gpu_layers        = 0,           \
    .n_workers           = 1,           \
    .max_queue_size      = 256,         \
    .default_max_tokens  = 256,         \
    .default_temperature = 0.0f,        \
    .seed                = -1,          \
    .preloaded_model     = NULL         \
}

/* ── Per-worker state ───────────────────────────────────────────────────── */
typedef struct {
    int                   worker_id;
    struct llama_context *ctx;
    void                 *engine;
} WorkerState;

/* ── Statistics ─────────────────────────────────────────────────────────── */
typedef struct {
    uint64_t jobs_submitted;
    uint64_t jobs_completed;
    uint64_t jobs_failed;
    double   total_exec_ms;
    double   total_queue_wait_ms;
    double   wall_time_ms;
} EngineStats;

/* ── Engine ─────────────────────────────────────────────────────────────── */
typedef struct {
    EngineConfig config;

    struct llama_model *model;
    int                 model_owned;  /* 1 = we loaded it, we free it */

    WorkerState     *worker_states;
    pthread_t       *worker_threads;
    volatile int     running;

    JobQueue        *queue;

    EngineStats      stats;
    pthread_mutex_t  stats_mutex;
    struct timespec  start_time;

    uint64_t         next_job_id;
    pthread_mutex_t  id_mutex;
} InferenceEngine;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */
InferenceEngine *engine_create(const EngineConfig *config);
int              engine_start(InferenceEngine *engine);
void             engine_stop(InferenceEngine *engine);
void             engine_destroy(InferenceEngine *engine);

/* ── Job submission ─────────────────────────────────────────────────────── */
InferenceJob *engine_submit_sync(InferenceEngine *engine,
                                 const char *prompt,
                                 int max_tokens, float temperature);
InferenceJob *engine_submit_async(InferenceEngine *engine,
                                  const char *prompt,
                                  int max_tokens, float temperature);

/* ── Stats ──────────────────────────────────────────────────────────────── */
void engine_get_stats(const InferenceEngine *engine, EngineStats *out);
void engine_print_stats(const InferenceEngine *engine);

#endif /* ENGINE_H */
