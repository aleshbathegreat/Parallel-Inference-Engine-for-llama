/*
 * main.c — Demonstration and stress-test of the Milestone 1 inference engine
 *
 * Phase 1: Sequential correctness check (3 prompts, one at a time)
 * Phase 2: Concurrent submission stress test (4 threads x 3 jobs)
 * Phase 3: Interactive user input (type prompts, 'quit' to exit)
 *
 * All prompts use TinyLLaMA's chat template format.
 *
 * Usage:
 *   ./inference_engine <model.gguf> [n_threads] [max_tokens]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "logger.h"
#include "engine.h"
#include "job.h"

/* ── Helper: wrap a plain question in TinyLLaMA chat template ───────────── */
#define CHAT(q) \
    "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n" q "</s>\n<|assistant|>\n"

/* ── Demo configuration ─────────────────────────────────────────────────── */
#define N_PRODUCER_THREADS  4
#define JOBS_PER_THREAD     3

static const char *CONCURRENT_PROMPTS[] = {
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
};
#define N_PROMPTS (int)(sizeof(CONCURRENT_PROMPTS) / sizeof(CONCURRENT_PROMPTS[0]))

/* ── Producer thread argument ───────────────────────────────────────────── */
typedef struct {
    InferenceEngine *engine;
    int              thread_id;
    int              max_tokens;
    int              jobs_done;
    int              jobs_failed;
} ProducerArg;

/* ── Producer thread function ───────────────────────────────────────────── */
static void *producer_fn(void *arg)
{
    ProducerArg     *parg   = (ProducerArg *)arg;
    InferenceEngine *engine = parg->engine;

    for (int i = 0; i < JOBS_PER_THREAD; i++) {
        int         prompt_idx = (parg->thread_id * JOBS_PER_THREAD + i) % N_PROMPTS;
        const char *prompt     = CONCURRENT_PROMPTS[prompt_idx];

        if (i % 2 == 0) {
            /* ── Synchronous ── */
            InferenceJob *job = engine_submit_sync(engine, prompt,
                                                   parg->max_tokens, 0.0f);
            if (job) {
                if (job->status == JOB_STATUS_DONE) {
                    printf("[Thread %d | Job #%llu | SYNC]\n"
                           "  Prompt : %s\n"
                           "  Output : %s\n"
                           "  Time   : queue=%.1f ms  exec=%.1f ms  total=%.1f ms\n\n",
                           parg->thread_id,
                           (unsigned long long)job->job_id,
                           job->prompt,
                           job->output,
                           job_queue_time_ms(job),
                           job_exec_time_ms(job),
                           job_total_time_ms(job));
                    parg->jobs_done++;
                } else {
                    printf("[Thread %d | Job #%llu | SYNC | ERROR: %s]\n",
                           parg->thread_id,
                           (unsigned long long)job->job_id,
                           job->error_msg);
                    parg->jobs_failed++;
                }
                job_destroy(job);
            } else {
                parg->jobs_failed++;
            }
        } else {
            /* ── Asynchronous ── */
            InferenceJob *job = engine_submit_async(engine, prompt,
                                                    parg->max_tokens, 0.0f);
            if (!job) { parg->jobs_failed++; continue; }

            usleep(1000);   /* 1 ms — simulate producer doing other work */
            job_wait(job);

            if (job->status == JOB_STATUS_DONE) {
                printf("[Thread %d | Job #%llu | ASYNC]\n"
                       "  Prompt : %s\n"
                       "  Output : %s\n"
                       "  Time   : queue=%.1f ms  exec=%.1f ms  total=%.1f ms\n\n",
                       parg->thread_id,
                       (unsigned long long)job->job_id,
                       job->prompt,
                       job->output,
                       job_queue_time_ms(job),
                       job_exec_time_ms(job),
                       job_total_time_ms(job));
                parg->jobs_done++;
            } else {
                printf("[Thread %d | Job #%llu | ASYNC | ERROR: %s]\n",
                       parg->thread_id,
                       (unsigned long long)job->job_id,
                       job->error_msg);
                parg->jobs_failed++;
            }
            job_destroy(job);
        }
    }

    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * main()
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> [n_threads] [max_tokens]\n"
                "Example: %s ./tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 4 64\n",
                argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    const char *model_path = argv[1];
    int         n_threads  = (argc >= 3) ? atoi(argv[2]) : 4;
    int         max_tokens = (argc >= 4) ? atoi(argv[3]) : 64;

    /* ── Logger ──────────────────────────────────────────────────────────── */
    logger_init("inference_engine.log", LOG_LEVEL_DEBUG);
    LOG_INFO("=== Parallel Inference Engine — Milestone 1 ===");
    LOG_INFO("Model: %s | threads: %d | max_tokens: %d",
             model_path, n_threads, max_tokens);

    /* ── Engine config ───────────────────────────────────────────────────── */
    EngineConfig config = ENGINE_CONFIG_DEFAULT;
    snprintf(config.model_path, sizeof(config.model_path), "%s", model_path);
    config.n_threads           = n_threads;
    config.n_ctx               = 2048;
    config.n_gpu_layers        = 0;
    config.default_max_tokens  = max_tokens;
    config.default_temperature = 0.0f;
    config.seed                = 42;

    /* ── Create and start ────────────────────────────────────────────────── */
    printf("Loading model, please wait...\n");
    InferenceEngine *engine = engine_create(&config);
    if (!engine) {
        LOG_ERROR("Failed to create engine.");
        logger_close();
        return EXIT_FAILURE;
    }

    if (engine_start(engine) != 0) {
        LOG_ERROR("Failed to start engine worker thread.");
        engine_destroy(engine);
        logger_close();
        return EXIT_FAILURE;
    }
    printf("Engine running.\n\n");

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 1: Sequential correctness check
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Phase 1: Sequential correctness check\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    static const char *seq_prompts[] = {
        CHAT("The capital of Japan is?"),
        CHAT("Explain what a binary search tree is."),
        CHAT("What are the three primary colours?"),
    };
    int n_seq = (int)(sizeof(seq_prompts) / sizeof(seq_prompts[0]));

    for (int i = 0; i < n_seq; i++) {
        InferenceJob *job = engine_submit_sync(engine, seq_prompts[i],
                                               max_tokens, 0.0f);
        if (!job) { LOG_ERROR("Sequential job %d failed to submit", i); continue; }

        printf("[Sequential %d/%d]\n", i + 1, n_seq);
        printf("  Status : %s\n",  job_status_str(job->status));
        if (job->status == JOB_STATUS_DONE)
            printf("  Output : %s\n",  job->output);
        else
            printf("  Error  : %s\n",  job->error_msg);
        printf("  Timing : queue=%.1f ms  exec=%.1f ms  total=%.1f ms\n\n",
               job_queue_time_ms(job),
               job_exec_time_ms(job),
               job_total_time_ms(job));

        job_destroy(job);
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 2: Concurrent submission stress test
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Phase 2: Concurrent submission (%d threads x %d jobs)\n",
           N_PRODUCER_THREADS, JOBS_PER_THREAD);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    pthread_t   threads[N_PRODUCER_THREADS];
    ProducerArg args[N_PRODUCER_THREADS];

    for (int t = 0; t < N_PRODUCER_THREADS; t++) {
        args[t].engine      = engine;
        args[t].thread_id   = t;
        args[t].max_tokens  = max_tokens;
        args[t].jobs_done   = 0;
        args[t].jobs_failed = 0;
        pthread_create(&threads[t], NULL, producer_fn, &args[t]);
    }

    int total_done   = 0;
    int total_failed = 0;
    for (int t = 0; t < N_PRODUCER_THREADS; t++) {
        pthread_join(threads[t], NULL);
        total_done   += args[t].jobs_done;
        total_failed += args[t].jobs_failed;
    }

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Concurrent test results:\n");
    printf("  Jobs succeeded : %d / %d\n", total_done,
           N_PRODUCER_THREADS * JOBS_PER_THREAD);
    printf("  Jobs failed    : %d\n", total_failed);
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    /* ═══════════════════════════════════════════════════════════════════════
     * Phase 3: Interactive user input
     * Your prompts go through the exact same queue/worker path as above.
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("Phase 3: Interactive mode\n");
    printf("Type a prompt and press Enter. Type 'quit' to exit.\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    char user_input[1024];
    char formatted[1200];

    while (1) {
        printf("You> ");
        fflush(stdout);

        if (!fgets(user_input, sizeof(user_input), stdin)) break;

        /* Strip trailing newline */
        size_t len = strlen(user_input);
        if (len > 0 && user_input[len - 1] == '\n')
            user_input[--len] = '\0';

        if (len == 0) continue;
        if (strcmp(user_input, "quit") == 0 ||
            strcmp(user_input, "exit") == 0) {
            printf("Exiting interactive mode.\n\n");
            break;
        }

        /* Wrap in chat template */
        snprintf(formatted, sizeof(formatted),
                 "<|system|>\nYou are a helpful assistant.</s>\n"
                 "<|user|>\n%s</s>\n<|assistant|>\n",
                 user_input);

        InferenceJob *job = engine_submit_sync(engine, formatted,
                                               max_tokens, 0.0f);
        if (!job) {
            printf("Error: failed to submit job.\n\n");
            continue;
        }

        if (job->status == JOB_STATUS_DONE) {
            printf("Bot> %s\n", job->output);
            printf("     [exec=%.1f ms  queue=%.1f ms]\n\n",
                   job_exec_time_ms(job),
                   job_queue_time_ms(job));
        } else {
            printf("Error: %s\n\n", job->error_msg);
        }

        job_destroy(job);
    }

    /* ── Shutdown ─────────────────────────────────────────────────────────── */
    engine_stop(engine);
    engine_print_stats(engine);
    engine_destroy(engine);

    logger_close();
    return (total_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
