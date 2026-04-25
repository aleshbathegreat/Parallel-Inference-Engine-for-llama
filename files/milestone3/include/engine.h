#ifndef ENGINE_H
#define ENGINE_H

/*
 * engine.h — High-performance Parallel Inference Engine (Milestone 2)
 *
 * Key optimisations over Milestone 1:
 *
 *  1. CPU AFFINITY — each worker thread is pinned to a specific CPU core
 *     using pthread_setaffinity_np().  This eliminates OS scheduler
 *     migrations between cores, reducing cache thrash and jitter.
 *
 *  2. LOCK-FREE STATS — job counters use __atomic builtins (CAS-free
 *     fetch-and-add) so workers never block each other updating stats.
 *
 *  3. BULK ASYNC SUBMISSION — all jobs are enqueued before any worker
 *     starts, keeping every worker busy from t=0 with no idle gaps.
 *
 *  4. TOKENS/SEC TRACKING — each job records n_tokens_generated so we
 *     can report aggregate throughput in tokens/second.
 *
 *  5. SEPARATE CONTEXTS — each worker owns its own llama_context so
 *     llama_decode() calls truly run in parallel with no locking.
 *
 *  Architecture:
 *
 *    All jobs → [ JobQueue ] → Worker-0 (core 0, ctx-0)
 *                            → Worker-1 (core 1, ctx-1)
 *                            → Worker-N (core N, ctx-N)
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
    int   n_ctx;
    int   n_threads;            /* CPU threads per worker context          */
    int   n_gpu_layers;
    int   n_workers;
    int   max_queue_size;
    int   default_max_tokens;
    float default_temperature;
    int   seed;
    int   use_cpu_affinity;     /* 1 = pin each worker to a specific core  */
    struct llama_model *preloaded_model;
} EngineConfig;

#define ENGINE_CONFIG_DEFAULT {     \
    .model_path          = "",      \
    .n_ctx               = 2048,    \
    .n_threads           = 1,       \
    .n_gpu_layers        = 0,       \
    .n_workers           = 1,       \
    .max_queue_size      = 512,     \
    .default_max_tokens  = 256,     \
    .default_temperature = 0.0f,    \
    .seed                = -1,      \
    .use_cpu_affinity    = 1,       \
    .preloaded_model     = NULL     \
}

/* ── Per-worker state ───────────────────────────────────────────────────── */
typedef struct {
    int                   worker_id;
    int                   cpu_core;       /* pinned core (-1 = not pinned) */
    struct llama_context *ctx;
    void                 *engine;
} WorkerState;

/* ── Lock-free statistics using atomics ─────────────────────────────────── */
typedef struct {
    /* All fields updated with __atomic_fetch_add — no mutex needed */
    uint64_t jobs_submitted;    /* set under id_mutex only at submit time  */
    uint64_t jobs_completed;
    uint64_t jobs_failed;
    /* These accumulate in nanoseconds to avoid float races */
    uint64_t total_exec_ns;
    uint64_t total_queue_ns;
    uint64_t total_tokens;
    /* Wall time set by engine_stop() */
    double   wall_time_ms;
} EngineStats;

/* ── Engine ─────────────────────────────────────────────────────────────── */
typedef struct {
    EngineConfig config;

    struct llama_model *model;
    int                 model_owned;

    WorkerState     *worker_states;
    pthread_t       *worker_threads;
    volatile int     running;

    JobQueue        *queue;

    EngineStats      stats;          /* updated atomically by workers      */
    pthread_mutex_t  stats_mutex;    /* only for wall_time_ms at stop time */
    struct timespec  start_time;

    uint64_t         next_job_id;
    pthread_mutex_t  id_mutex;
} InferenceEngine;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */
InferenceEngine *engine_create(const EngineConfig *config);
int              engine_start(InferenceEngine *engine);
void             engine_stop(InferenceEngine *engine);
void             engine_destroy(InferenceEngine *engine);

/* ── Submission ─────────────────────────────────────────────────────────── */
InferenceJob *engine_submit_async(InferenceEngine *engine,
                                  const char *prompt,
                                  int max_tokens, float temperature);
InferenceJob *engine_submit_sync(InferenceEngine *engine,
                                 const char *prompt,
                                 int max_tokens, float temperature);

/* ── Stats ──────────────────────────────────────────────────────────────── */
void engine_get_stats(const InferenceEngine *engine, EngineStats *out);
void engine_print_stats(const InferenceEngine *engine);

#endif /* ENGINE_H */
