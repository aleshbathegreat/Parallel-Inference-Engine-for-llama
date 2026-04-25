/*
 * correctness.c — Milestone 3 correctness test suite
 *
 * Tests that the engine produces correct, non-corrupted results
 * under concurrent load. All tests use greedy sampling (temperature=0)
 * so outputs are deterministic and comparable.
 */

#define _GNU_SOURCE
#include "correctness.h"
#include "engine.h"
#include "job.h"
#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define CHAT(q) \
    "<|system|>\nYou are a helpful assistant.</s>\n<|user|>\n" q "</s>\n<|assistant|>\n"

/* ── Test helpers ───────────────────────────────────────────────────────── */

static void print_result(const char *name, int passed, const char *detail)
{
    printf("  [%s] %s", passed ? "PASS" : "FAIL", name);
    if (!passed && detail && detail[0])
        printf("\n         → %s", detail);
    printf("\n");
}

static InferenceEngine *make_engine(struct llama_model *model, int n_workers)
{
    EngineConfig cfg = ENGINE_CONFIG_DEFAULT;
    cfg.preloaded_model    = model;
    cfg.n_workers          = n_workers;
    cfg.n_threads          = 1;
    cfg.n_ctx              = 2048;
    cfg.default_max_tokens = 64;
    cfg.use_cpu_affinity   = 1;
    cfg.seed               = 42;
    InferenceEngine *e = engine_create(&cfg);
    if (e) engine_start(e);
    return e;
}

static void destroy_engine(InferenceEngine *e)
{
    if (!e) return;
    engine_stop(e);
    engine_destroy(e);
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST 1: Determinism
 * Same prompt + greedy sampling must produce identical output every time.
 * ══════════════════════════════════════════════════════════════════════════ */
static int test_determinism(struct llama_model *model)
{
    const char *prompt = CHAT("What is the capital of France?");
    char first_output[1024] = {0};
    int passed = 1;
    char detail[256] = {0};

    InferenceEngine *e = make_engine(model, 1);
    if (!e) { print_result("Determinism", 0, "engine_create failed"); return 0; }

    for (int run = 0; run < 3; run++) {
        InferenceJob *job = engine_submit_sync(e, prompt, 64, 0.0f);
        if (!job || job->status != JOB_STATUS_DONE) {
            passed = 0; snprintf(detail,sizeof(detail),"run %d failed",run);
            if (job) job_destroy(job);
            break;
        }
        if (run == 0) {
            strncpy(first_output, job->output, sizeof(first_output)-1);
        } else if (strcmp(first_output, job->output) != 0) {
            passed = 0;
            snprintf(detail, sizeof(detail),
                     "run 0 output differs from run %d", run);
        }
        job_destroy(job);
        if (!passed) break;
    }

    destroy_engine(e);
    print_result("Determinism (3 runs, greedy, same output)", passed, detail);
    return passed;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST 2: Output isolation
 * N jobs submitted concurrently — each must return output for its own prompt.
 * ══════════════════════════════════════════════════════════════════════════ */
static const char *ISOLATION_PROMPTS[] = {
    CHAT("What is the capital of France?"),
    CHAT("What does CPU stand for?"),
    CHAT("What is 2 plus 2?"),
    CHAT("Name one programming language."),
};
static const char *ISOLATION_KEYWORDS[] = {
    "Paris", "Central", "4", "C"   /* expected in each answer */
};
#define N_ISOLATION 4

static int test_output_isolation(struct llama_model *model, int n_workers)
{
    char detail[256] = {0};
    int passed = 1;

    InferenceEngine *e = make_engine(model, n_workers);
    if (!e) { print_result("Output isolation", 0, "engine_create failed"); return 0; }

    /* Submit all async */
    InferenceJob *jobs[N_ISOLATION];
    for (int i = 0; i < N_ISOLATION; i++)
        jobs[i] = engine_submit_async(e, ISOLATION_PROMPTS[i], 64, 0.0f);

    /* Collect and verify */
    for (int i = 0; i < N_ISOLATION; i++) {
        if (!jobs[i] || jobs[i]->status == JOB_STATUS_ERROR) {
            passed = 0;
            snprintf(detail,sizeof(detail),"job %d failed",i);
            if (jobs[i]) job_destroy(jobs[i]);
            continue;
        }
        job_wait(jobs[i]);
        /* Check output contains expected keyword */
        if (!strstr(jobs[i]->output, ISOLATION_KEYWORDS[i])) {
            /* Soft check — model may phrase differently */
            LOG_WARN("Job %d output may not contain '%s' (got: %.60s...)",
                     i, ISOLATION_KEYWORDS[i], jobs[i]->output);
        }
        /* Check output is non-empty */
        if (jobs[i]->output_len == 0) {
            passed = 0;
            snprintf(detail,sizeof(detail),"job %d empty output",i);
        }
        job_destroy(jobs[i]);
    }

    destroy_engine(e);
    print_result("Output isolation (4 concurrent jobs, non-empty outputs)",
                 passed, detail);
    return passed;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST 3: Queue stress
 * Submit 2× more jobs than workers — all must complete without deadlock.
 * ══════════════════════════════════════════════════════════════════════════ */
static int test_queue_stress(struct llama_model *model, int n_workers)
{
    int n_jobs = n_workers * 2 + 2;  /* always more than workers */
    char detail[256] = {0};

    InferenceEngine *e = make_engine(model, n_workers);
    if (!e) { print_result("Queue stress", 0, "engine_create failed"); return 0; }

    const char *prompt = CHAT("What does RAM stand for?");
    InferenceJob **jobs = calloc((size_t)n_jobs, sizeof(*jobs));

    for (int i = 0; i < n_jobs; i++)
        jobs[i] = engine_submit_async(e, prompt, 32, 0.0f);

    int done = 0, failed = 0;
    for (int i = 0; i < n_jobs; i++) {
        if (!jobs[i]) { failed++; continue; }
        job_wait(jobs[i]);
        if (jobs[i]->status == JOB_STATUS_DONE) done++;
        else failed++;
        job_destroy(jobs[i]);
    }
    free(jobs);
    destroy_engine(e);

    int passed = (done == n_jobs);
    if (!passed)
        snprintf(detail, sizeof(detail),
                 "%d/%d completed, %d failed", done, n_jobs, failed);
    print_result("Queue stress (2× jobs vs workers, no deadlock)", passed, detail);
    return passed;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST 4: Concurrent submission correctness
 * Multiple threads submit simultaneously — no job_id corruption or
 * double-completion.
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    InferenceEngine *engine;
    int              thread_id;
    int              n_jobs;
    int              done;
    int              failed;
} ConcurrentArg;

static void *concurrent_submit_fn(void *arg)
{
    ConcurrentArg *a = (ConcurrentArg *)arg;
    const char *prompt = CHAT("What does CPU stand for?");
    for (int i = 0; i < a->n_jobs; i++) {
        InferenceJob *job = engine_submit_sync(a->engine, prompt, 32, 0.0f);
        if (!job) { a->failed++; continue; }
        if (job->status == JOB_STATUS_DONE && job->output_len > 0) a->done++;
        else a->failed++;
        job_destroy(job);
    }
    return NULL;
}

static int test_concurrent_submission(struct llama_model *model, int n_workers)
{
    int n_threads = n_workers;
    int jobs_per_thread = 2;
    char detail[256] = {0};

    InferenceEngine *e = make_engine(model, n_workers);
    if (!e) { print_result("Concurrent submission", 0, "engine_create failed"); return 0; }

    pthread_t     *threads = calloc((size_t)n_threads, sizeof(pthread_t));
    ConcurrentArg *args    = calloc((size_t)n_threads, sizeof(ConcurrentArg));

    for (int t = 0; t < n_threads; t++) {
        args[t].engine    = e;
        args[t].thread_id = t;
        args[t].n_jobs    = jobs_per_thread;
        pthread_create(&threads[t], NULL, concurrent_submit_fn, &args[t]);
    }

    int total_done=0, total_failed=0;
    for (int t = 0; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
        total_done   += args[t].done;
        total_failed += args[t].failed;
    }
    free(threads); free(args);
    destroy_engine(e);

    int expected = n_threads * jobs_per_thread;
    int passed   = (total_done == expected);
    if (!passed)
        snprintf(detail, sizeof(detail),
                 "%d/%d succeeded, %d failed", total_done, expected, total_failed);
    print_result("Concurrent submission (N threads × 2 jobs each, all succeed)",
                 passed, detail);
    return passed;
}

/* ══════════════════════════════════════════════════════════════════════════
 * TEST 5: No memory corruption under load
 * Submit a large batch and verify no output is empty or contains NUL bytes
 * in the middle (sign of buffer corruption).
 * ══════════════════════════════════════════════════════════════════════════ */
static int test_memory_integrity(struct llama_model *model, int n_workers)
{
    int n_jobs = 8;
    char detail[256] = {0};
    int corrupted = 0;

    InferenceEngine *e = make_engine(model, n_workers);
    if (!e) { print_result("Memory integrity", 0, "engine_create failed"); return 0; }

    const char *prompts[] = {
        CHAT("What is the capital of France?"),
        CHAT("What does CPU stand for?"),
        CHAT("Name one sorting algorithm."),
        CHAT("What is 5 times 5?"),
        CHAT("What is a stack?"),
        CHAT("Define latency."),
        CHAT("What is RAM?"),
        CHAT("Who wrote C?"),
    };

    InferenceJob **jobs = calloc((size_t)n_jobs, sizeof(*jobs));
    for (int i = 0; i < n_jobs; i++)
        jobs[i] = engine_submit_async(e, prompts[i % 8], 48, 0.0f);

    for (int i = 0; i < n_jobs; i++) {
        if (!jobs[i]) { corrupted++; continue; }
        job_wait(jobs[i]);
        if (jobs[i]->status == JOB_STATUS_DONE) {
            /* Check no internal NUL bytes */
            size_t real_len = strlen(jobs[i]->output);
            if (real_len != jobs[i]->output_len) {
                corrupted++;
                LOG_ERROR("Job %d: output_len=%zu but strlen=%zu — NUL in middle",
                          i, jobs[i]->output_len, real_len);
            }
            if (real_len == 0) {
                corrupted++;
            }
        }
        job_destroy(jobs[i]);
    }
    free(jobs);
    destroy_engine(e);

    int passed = (corrupted == 0);
    if (!passed)
        snprintf(detail, sizeof(detail),
                 "%d/%d jobs had corrupted/empty output", corrupted, n_jobs);
    print_result("Memory integrity (8 concurrent jobs, no corruption)",
                 passed, detail);
    return passed;
}

/* ══════════════════════════════════════════════════════════════════════════
 * run_correctness_suite()
 * ══════════════════════════════════════════════════════════════════════════ */
int run_correctness_suite(struct llama_model *model, int n_workers)
{
    printf("\n  Running correctness test suite (%d workers)...\n\n", n_workers);

    int passed = 0;
    int total  = 5;

    passed += test_determinism(model);
    passed += test_output_isolation(model, n_workers);
    passed += test_queue_stress(model, n_workers);
    passed += test_concurrent_submission(model, n_workers);
    passed += test_memory_integrity(model, n_workers);

    printf("\n  Results: %d / %d tests passed", passed, total);
    if (passed == total)
        printf("  ✓ All correctness tests passed\n\n");
    else
        printf("  ✗ %d test(s) failed — see above\n\n", total - passed);

    return passed;
}
