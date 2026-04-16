/*
 * main.c — Milestone 2: Parallel Execution Benchmark
 *
 * Clean output: llama.cpp internal logs are suppressed.
 * Only our own logger and the benchmark table are shown.
 *
 * Phase 1 — Serial baseline   (1 worker)
 * Phase 2 — Parallel          (N workers)
 * Phase 3 — Comparison table
 * Phase 4 — Interactive chat
 *
 * Usage:
 *   ./inference_engine <model.gguf> [n_workers] [max_tokens]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdarg.h>
#include <unistd.h>

#include "logger.h"
#include "engine.h"
#include "job.h"
#include "llama.h"
#include "ggml.h"

/* ── Suppress all llama.cpp / ggml internal output ─────────────────────── */
static void silent_log(enum ggml_log_level level, const char *text, void *ud)
{
    (void)level; (void)text; (void)ud;  /* discard everything */
}

/* ── Chat template ──────────────────────────────────────────────────────── */
#define CHAT(q) \
    "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n" q "</s>\n<|assistant|>\n"

/* ── Benchmark prompts ──────────────────────────────────────────────────── */
static const char *BENCH_PROMPTS[] = {
    CHAT("What is the capital of France?"),
    CHAT("Explain what a mutex is in one sentence."),
    CHAT("Write a haiku about parallel computing."),
    CHAT("What is 17 times 13?"),
    CHAT("Name three programming languages invented before 1980."),
    CHAT("What does CPU stand for?"),
    CHAT("Describe a stack data structure briefly."),
    CHAT("What is the boiling point of water in Celsius?"),
    CHAT("Who wrote the C programming language?"),
    CHAT("What is an operating system?"),
    CHAT("Define latency in computer networks."),
    CHAT("What is a thread in operating systems?"),
    CHAT("What is a binary search tree?"),
    CHAT("Explain what RAM stands for."),
    CHAT("What is the difference between a process and a thread?"),
    CHAT("Name two sorting algorithms."),
};
#define N_BENCH (int)(sizeof(BENCH_PROMPTS)/sizeof(BENCH_PROMPTS[0]))
#define N_PRODUCERS 4

/* ── Producer thread ────────────────────────────────────────────────────── */
typedef struct {
    InferenceEngine *engine;
    int              start_idx;
    int              count;
    int              max_tokens;
    int              done;
    int              failed;
} ProducerArg;

static void *producer_fn(void *arg)
{
    ProducerArg *p = (ProducerArg *)arg;
    for (int i = 0; i < p->count; i++) {
        int idx = (p->start_idx + i) % N_BENCH;
        InferenceJob *job = engine_submit_sync(p->engine, BENCH_PROMPTS[idx],
                                               p->max_tokens, 0.0f);
        if (!job) { p->failed++; continue; }

        if (job->status == JOB_STATUS_DONE) {
            /* Extract just the user question from the chat template for display */
            const char *q_start = strstr(BENCH_PROMPTS[idx], "<|user|>\n");
            const char *q_end   = strstr(BENCH_PROMPTS[idx], "</s>\n<|assistant|>");
            char question[256]  = "(unknown)";
            if (q_start && q_end) {
                q_start += strlen("<|user|>\n");
                size_t qlen = (size_t)(q_end - q_start);
                if (qlen >= sizeof(question)) qlen = sizeof(question) - 1;
                memcpy(question, q_start, qlen);
                question[qlen] = '\0';
            }

            /* Strip trailing </s> from output */
            char *eos = strstr(job->output, "</s>");
            if (eos) *eos = '\0';

            printf("  Q: %s\n  A: %s\n     [exec=%.0f ms  queue=%.0f ms]\n\n",
                   question, job->output,
                   job_exec_time_ms(job), job_queue_time_ms(job));
            p->done++;
        } else {
            p->failed++;
        }
        job_destroy(job);
    }
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * run_benchmark()
 * ═══════════════════════════════════════════════════════════════════════════ */
static EngineStats run_benchmark(struct llama_model *model,
                                 int n_workers, int max_tokens,
                                 int threads_per_worker)
{
    EngineStats empty = {0};

    EngineConfig cfg = ENGINE_CONFIG_DEFAULT;
    cfg.preloaded_model    = model;
    cfg.n_workers          = n_workers;
    cfg.n_threads          = threads_per_worker;
    cfg.n_ctx              = 2048;
    cfg.default_max_tokens = max_tokens;
    cfg.seed               = 42;

    InferenceEngine *engine = engine_create(&cfg);
    if (!engine) { fprintf(stderr, "engine_create failed\n"); return empty; }
    engine_start(engine);

    printf("  Starting %d jobs...\n\n", N_BENCH);

    /* Divide prompts among producers */
    int per = N_BENCH / N_PRODUCERS, rem = N_BENCH % N_PRODUCERS;
    pthread_t   threads[N_PRODUCERS];
    ProducerArg args[N_PRODUCERS];
    int offset = 0;

    for (int t = 0; t < N_PRODUCERS; t++) {
        args[t] = (ProducerArg){engine, offset, per+(t<rem?1:0), max_tokens, 0, 0};
        offset += args[t].count;
        pthread_create(&threads[t], NULL, producer_fn, &args[t]);
    }

    int total_done = 0, total_failed = 0;
    for (int t = 0; t < N_PRODUCERS; t++) {
        pthread_join(threads[t], NULL);
        total_done   += args[t].done;
        total_failed += args[t].failed;
    }

    engine_stop(engine);

    EngineStats stats;
    engine_get_stats(engine, &stats);
    engine_destroy(engine);

    if (total_failed > 0)
        printf("  Warning: %d job(s) failed\n", total_failed);

    return stats;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> [n_workers] [max_tokens]\n"
                "Example: %s ./tinyllama.gguf 2 128\n",
                argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    int         n_workers  = (argc >= 3) ? atoi(argv[2]) : 2;
    int         max_tokens = (argc >= 4) ? atoi(argv[3]) : 128;
    if (n_workers < 1) n_workers = 1;
    if (n_workers > 8) n_workers = 8;

    int total_threads      = 4;
    int threads_per_worker = (total_threads / n_workers) < 1
                             ? 1 : (total_threads / n_workers);

    /* Suppress ALL llama.cpp / ggml internal output */
    llama_log_set(silent_log, NULL);

    logger_init("inference_engine.log", LOG_LEVEL_DEBUG);

    /* ── Banner ──────────────────────────────────────────────────────────── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║      Parallel Inference Engine  —  Milestone 2       ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Model      : %-38s ║\n", model_path);
    printf("║  Prompts    : %-38d ║\n", N_BENCH);
    printf("║  Max tokens : %-38d ║\n", max_tokens);
    printf("║  Workers    : 1 (serial)  →  %-25d ║\n", n_workers);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* ── Load model once ─────────────────────────────────────────────────── */
    printf("  Loading model... ");
    fflush(stdout);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(model_path, mp);
    if (!model) {
        printf("FAILED\n");
        fprintf(stderr, "Cannot load '%s'\n", model_path);
        llama_backend_free();
        logger_close();
        return EXIT_FAILURE;
    }
    printf("done.\n\n");

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 1 — Serial
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 1 — Serial baseline  (1 worker, %d threads)\n", total_threads);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    EngineStats serial = run_benchmark(model, 1, max_tokens,
                                       total_threads);

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 2 — Parallel
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 2 — Parallel execution  (%d workers, %d thread(s) each)\n",
           n_workers, threads_per_worker);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    EngineStats parallel = run_benchmark(model, n_workers, max_tokens,
                                         threads_per_worker);

    /* Interactive phase follows — summary printed after */

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 3 — Interactive chat (results table shown after this)
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 3 — Interactive chat  (%d workers)\n", n_workers);
    printf("  Type a prompt and press Enter.  'quit' to exit.\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    EngineConfig icfg = ENGINE_CONFIG_DEFAULT;
    icfg.preloaded_model    = model;
    icfg.n_workers          = n_workers;
    icfg.n_threads          = threads_per_worker;
    icfg.n_ctx              = 2048;
    icfg.default_max_tokens = max_tokens;
    icfg.seed               = 42;

    InferenceEngine *ie = engine_create(&icfg);
    if (!ie) { fprintf(stderr, "interactive engine failed\n"); goto summary; }
    engine_start(ie);

    char user_input[1024], formatted[1200];

    while (1) {
        printf("\nYou> "); fflush(stdout);
        if (!fgets(user_input, sizeof(user_input), stdin)) break;

        size_t len = strlen(user_input);
        if (len > 0 && user_input[len-1] == '\n') user_input[--len] = '\0';
        if (len == 0) continue;
        if (!strcmp(user_input, "quit") || !strcmp(user_input, "exit")) {
            printf("\nGoodbye!\n\n"); break;
        }

        snprintf(formatted, sizeof(formatted),
                 "<|system|>\nYou are a helpful assistant.</s>\n"
                 "<|user|>\n%s</s>\n<|assistant|>\n", user_input);

        InferenceJob *job = engine_submit_sync(ie, formatted, max_tokens, 0.0f);
        if (!job) { printf("(submit failed)\n\n"); continue; }

        /* Strip trailing </s> */
        char *eos = strstr(job->output, "</s>");
        if (eos) *eos = '\0';

        if (job->status == JOB_STATUS_DONE) {
            printf("\n  ── Result ──────────────────────────────────────\n");
            printf("  You : %s\n", user_input);
            printf("  Bot : %s\n", job->output);
            printf("        [exec=%.0f ms  queue=%.0f ms]\n", 
                   job_exec_time_ms(job), job_queue_time_ms(job));
            printf("  ────────────────────────────────────────────────\n");
        } else {
            printf("Error: %s\n\n", job->error_msg);
        }

        job_destroy(job);
    }

    engine_stop(ie);
    engine_destroy(ie);

summary:
    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 4 — Benchmark Results (shown after interactive so it's the last
     *            thing you see — easy to screenshot for your report)
     * ═══════════════════════════════════════════════════════════════════════ */
    {
        double s_wall  = serial.wall_time_ms,   p_wall  = parallel.wall_time_ms;
        double s_exec  = serial.jobs_completed  > 0 ? serial.total_exec_ms            / serial.jobs_completed   : 0;
        double p_exec  = parallel.jobs_completed > 0 ? parallel.total_exec_ms          / parallel.jobs_completed : 0;
        double s_queue = serial.jobs_completed  > 0 ? serial.total_queue_wait_ms       / serial.jobs_completed   : 0;
        double p_queue = parallel.jobs_completed > 0 ? parallel.total_queue_wait_ms    / parallel.jobs_completed : 0;
        double s_tput  = s_wall > 0 ? serial.jobs_completed   / (s_wall/1000.0) : 0;
        double p_tput  = p_wall > 0 ? parallel.jobs_completed / (p_wall/1000.0) : 0;
        double speedup = s_wall > 0 && p_wall > 0 ? s_wall / p_wall : 0;
        double q_drop  = s_queue > 0 ? (1.0 - p_queue/s_queue)*100.0 : 0;

        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Phase 4 — Benchmark Summary\n");
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

        printf("  ┌────────────────────────────┬──────────────┬──────────────┐\n");
        printf("  │ Metric                     │  Serial (1w) │Parallel (%dw) │\n", n_workers);
        printf("  ├────────────────────────────┼──────────────┼──────────────┤\n");
        printf("  │ Wall-clock time      (ms)  │ %12.0f │ %12.0f │\n", s_wall, p_wall);
        printf("  │ Avg exec time        (ms)  │ %12.0f │ %12.0f │\n", s_exec, p_exec);
        printf("  │ Avg queue wait       (ms)  │ %12.0f │ %12.0f │\n", s_queue, p_queue);
        printf("  │ Throughput        (jobs/s) │ %12.3f │ %12.3f │\n", s_tput, p_tput);
        printf("  ├────────────────────────────┼──────────────┴──────────────┤\n");
        printf("  │ Speedup (wall time)        │       %.2fx                   │\n", speedup);
        printf("  │ Queue wait reduction       │       %.1f%%                  │\n", q_drop);
        printf("  └────────────────────────────┴─────────────────────────────┘\n\n");

        if (speedup >= 1.5)
            printf("  ✓  Parallel is %.1fx faster — parallelisation is clearly justified.\n\n", speedup);
        else if (speedup >= 1.0)
            printf("  ~  Modest speedup (%.1fx). Bottleneck is CPU saturation —\n"
                   "     %d workers share %d cores. More cores = much higher speedup.\n\n",
                   speedup, n_workers, total_threads);
        else
            printf("  !  Parallel was slower — too many workers for available cores.\n\n");

        printf("  Key insight: avg queue wait dropped from %.0f ms → %.0f ms (%.0f%% reduction).\n"
               "  Jobs no longer wait behind each other — that is the value of parallelisation.\n\n",
               s_queue, p_queue, q_drop);
    }

cleanup:
    llama_model_free(model);
    llama_backend_free();
    logger_close();
    return EXIT_SUCCESS;
}
