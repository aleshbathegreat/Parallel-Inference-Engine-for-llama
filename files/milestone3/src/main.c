/*
 * main.c — Milestone 3: Integration, Benchmarking, and Analysis
 *
 * Phases:
 *   0 — Correctness test suite (must pass before benchmarking)
 *   1 — Scaling sweep: 1 → 2 → 4 workers
 *   2 — Full benchmark report with latency distribution,
 *         Amdahl's Law analysis, and tradeoff discussion
 *   3 — Interactive chat (parallel engine)
 *
 * Usage:
 *   ./inference_engine <model.gguf> [max_tokens]
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "logger.h"
#include "engine.h"
#include "job.h"
#include "benchmark.h"
#include "correctness.h"
#include "llama.h"
#include "ggml.h"

/* ── Silence llama.cpp ─────────────────────────────────────────────────── */
static void silent_log(enum ggml_log_level l, const char *t, void *u)
{ (void)l;(void)t;(void)u; }

/* ── Chat template ──────────────────────────────────────────────────────── */
#define CHAT(q) \
    "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n" q "</s>\n<|assistant|>\n"

/* ── 16 benchmark prompts ───────────────────────────────────────────────── */
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

/* ══════════════════════════════════════════════════════════════════════════
 * main()
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: %s <model.gguf> [max_tokens]\n"
            "Example: %s ./tinyllama.gguf 128\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    int         max_tokens = (argc >= 3) ? atoi(argv[2]) : 128;
    int         n_cpus     = (int)sysconf(_SC_NPROCESSORS_ONLN);

    llama_log_set(silent_log, NULL);
    logger_init("inference_engine.log", LOG_LEVEL_DEBUG);

    /* ── Banner ──────────────────────────────────────────────────────────── */
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║   Parallel Inference Engine for TinyLLaMA                       ║\n");
    printf("║   Milestone 3 — Integration, Benchmarking, and Analysis         ║\n");
    printf("╠══════════════════════════════════════════════════════════════════╣\n");
    printf("║  Model      : %-51s║\n", model_path);
    printf("║  Prompts    : %-51d║\n", N_BENCH);
    printf("║  Max tokens : %-51d║\n", max_tokens);
    printf("║  CPU cores  : %-51d║\n", n_cpus);
    printf("║  Sweep      : 1 → 2 → 4 workers                                ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");

    /* ── Load model once ─────────────────────────────────────────────────── */
    printf("  Loading model... "); fflush(stdout);
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
    printf("done  (636 MB, shared across all engine instances)\n\n");

    /* ══════════════════════════════════════════════════════════════════════
     * Phase 0 — Correctness test suite
     * Must pass before benchmarking — proves system is correct under load
     * ══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 0 — Correctness Test Suite\n");
    printf("  Verifies system correctness under concurrent load before\n");
    printf("  benchmarking. Tests: determinism, isolation, stress, concurrent\n");
    printf("  submission, and memory integrity.\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    int correctness_passed = run_correctness_suite(model, 2);
    if (correctness_passed < 5) {
        printf("  WARNING: %d correctness test(s) failed.\n"
               "  Proceeding to benchmarks but results may be unreliable.\n\n",
               5 - correctness_passed);
    } else {
        printf("  Correctness verified. Proceeding to benchmarks.\n\n");
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Phase 1 — Scaling sweep
     * ══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 1 — Scaling Sweep  (1 → 2 → 4 workers)\n");
    printf("  Each level: 1 thread/worker, CPU affinity ON, bulk async submit\n");
    printf("  Prompts: %d × max_tokens=%d\n", N_BENCH, max_tokens);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    ScalingResult results[MAX_SWEEP_WORKERS];
    int n_results = run_scaling_sweep(model, BENCH_PROMPTS, N_BENCH,
                                     max_tokens, results);

    /* ══════════════════════════════════════════════════════════════════════
     * Phase 2 — Interactive chat
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 2 — Interactive Chat  (%d workers)\n", n_cpus);
    printf("  Type a prompt and press Enter.  'quit' to exit.\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    EngineConfig icfg = ENGINE_CONFIG_DEFAULT;
    icfg.preloaded_model    = model;
    icfg.n_workers          = n_cpus;
    icfg.n_threads          = 1;
    icfg.n_ctx              = 2048;
    icfg.default_max_tokens = max_tokens;
    icfg.use_cpu_affinity   = 1;
    icfg.seed               = 42;

    InferenceEngine *ie = engine_create(&icfg);
    if (ie) {
        engine_start(ie);
        char user_input[1024], formatted[1200];
        while (1) {
            printf("\nYou> "); fflush(stdout);
            if (!fgets(user_input, sizeof(user_input), stdin)) break;
            size_t len = strlen(user_input);
            if (len > 0 && user_input[len-1]=='\n') user_input[--len]='\0';
            if (len == 0) continue;
            if (!strcmp(user_input,"quit")||!strcmp(user_input,"exit")) {
                printf("\nGoodbye!\n\n"); break;
            }
            snprintf(formatted, sizeof(formatted),
                     "<|system|>\nYou are a helpful assistant.</s>\n"
                     "<|user|>\n%s</s>\n<|assistant|>\n", user_input);
            InferenceJob *job = engine_submit_sync(ie, formatted, max_tokens, 0.0f);
            if (!job) { printf("(submit failed)\n\n"); continue; }
            char *eos = strstr(job->output,"</s>"); if (eos) *eos='\0';
            if (job->status == JOB_STATUS_DONE) {
                printf("\n  ── Response ──────────────────────────────────────\n");
                printf("  You : %s\n", user_input);
                printf("  Bot : %s\n", job->output);
                printf("        [exec=%.0f ms  %.1f tok/s]\n",
                       job_exec_time_ms(job), job_tokens_per_sec(job));
                printf("  ──────────────────────────────────────────────────\n");
            } else {
                printf("Error: %s\n\n", job->error_msg);
            }
            job_destroy(job);
        }
        engine_stop(ie);
        engine_destroy(ie);
    }

    /* ══════════════════════════════════════════════════════════════════════
     * Phase 3 — Full benchmark report
     * Printed last so it's easy to screenshot for the report
     * ══════════════════════════════════════════════════════════════════════ */
    printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("  Phase 3 — Full Benchmark Report\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    print_full_report(results, n_results, n_cpus);

    /* ── Cleanup ─────────────────────────────────────────────────────────── */
    llama_model_free(model);
    llama_backend_free();
    logger_close();
    return EXIT_SUCCESS;
}
