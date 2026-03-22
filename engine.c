/*
 * engine.c — Core inference orchestration layer
 *
 * API verified against llama.cpp build in ~/Downloads/files/llama.cpp
 *
 * Key API calls used:
 *   llama_model_load_from_file()     — load model
 *   llama_model_free()               — free model
 *   llama_init_from_model()          — create context
 *   llama_free()                     — free context
 *   llama_model_get_vocab()          — get vocab handle
 *   llama_vocab_n_tokens()           — vocab size
 *   llama_tokenize(vocab, ...)       — tokenise prompt
 *   llama_get_memory() +
 *   llama_memory_clear(mem, false)   — clear KV cache
 *   llama_batch_init/free()          — batch management
 *   llama_decode()                   — run forward pass
 *   llama_get_logits_ith()           — get logits
 *   llama_vocab_is_eog()             — end-of-generation check
 *   llama_token_to_piece(vocab, ...) — detokenise
 */

#include "engine.h"
#include "logger.h"

#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

/* ── Token sampling ─────────────────────────────────────────────────────── */

static llama_token greedy_sample(const float *logits, int n_vocab)
{
    llama_token best       = 0;
    float       best_logit = -FLT_MAX;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > best_logit) {
            best_logit = logits[i];
            best       = i;
        }
    }
    return best;
}

typedef struct { int idx; float prob; } TokenProb;

static int tp_cmp_desc(const void *a, const void *b)
{
    float da = ((const TokenProb *)a)->prob;
    float db = ((const TokenProb *)b)->prob;
    return (da < db) - (da > db);
}

static llama_token temperature_sample(const float *logits, int n_vocab,
                                      float temperature, unsigned int *seed)
{
    float *scaled = malloc((size_t)n_vocab * sizeof(float));
    if (!scaled) return greedy_sample(logits, n_vocab);

    float max_l = -FLT_MAX;
    for (int i = 0; i < n_vocab; i++) {
        float v = logits[i] / temperature;
        if (v > max_l) max_l = v;
        scaled[i] = v;
    }

    float sum = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        scaled[i] = expf(scaled[i] - max_l);
        sum += scaled[i];
    }
    for (int i = 0; i < n_vocab; i++) scaled[i] /= sum;

    TokenProb *tp = malloc((size_t)n_vocab * sizeof(TokenProb));
    if (!tp) { free(scaled); return greedy_sample(logits, n_vocab); }

    for (int i = 0; i < n_vocab; i++) { tp[i].idx = i; tp[i].prob = scaled[i]; }
    qsort(tp, (size_t)n_vocab, sizeof(TokenProb), tp_cmp_desc);
    free(scaled);

    float cumsum = 0.0f;
    int   cutoff = n_vocab;
    for (int i = 0; i < n_vocab; i++) {
        cumsum += tp[i].prob;
        if (cumsum >= 0.9f) { cutoff = i + 1; break; }
    }

    float norm = 0.0f;
    for (int i = 0; i < cutoff; i++) norm += tp[i].prob;

    float r   = ((float)(rand_r(seed) & 0x7FFF)) / (float)0x8000 * norm;
    float acc = 0.0f;
    llama_token chosen = tp[0].idx;
    for (int i = 0; i < cutoff; i++) {
        acc += tp[i].prob;
        if (r <= acc) { chosen = tp[i].idx; break; }
    }

    free(tp);
    return chosen;
}

/* ── Batch helpers ──────────────────────────────────────────────────────── */

static void batch_clear(struct llama_batch *batch)
{
    batch->n_tokens = 0;
}

static void batch_add_token(struct llama_batch *batch,
                             llama_token token, int pos,
                             int seq_id, int want_logits)
{
    int n = batch->n_tokens;
    batch->token    [n]    = token;
    batch->pos      [n]    = pos;
    batch->n_seq_id [n]    = 1;
    batch->seq_id   [n][0] = seq_id;
    batch->logits   [n]    = (int8_t)want_logits;
    batch->n_tokens++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Core inference routine
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Returns 0 on success, -1 on error.
 * Does NOT call job_mark_done/error — the worker thread does that AFTER
 * capturing timing stats, preventing a use-after-free race. */
static int run_inference(InferenceEngine *engine, InferenceJob *job)
{
    struct llama_model       *model = engine->model;
    struct llama_context     *ctx   = engine->ctx;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_ctx_max              = (int)llama_n_ctx(ctx);
    const int n_vocab                = llama_vocab_n_tokens(vocab);

    /* ── 1. Tokenise prompt ─────────────────────────────────────────────── */
    llama_token *prompt_tokens = malloc((size_t)n_ctx_max * sizeof(llama_token));
    if (!prompt_tokens) {
        strncpy(job->error_msg, "tokenise: out of memory",
                sizeof(job->error_msg) - 1);
        return -1;
    }

    int n_prompt = llama_tokenize(
        vocab,
        job->prompt, (int32_t)strlen(job->prompt),
        prompt_tokens, n_ctx_max,
        true,
        false
    );

    if (n_prompt < 0) {
        free(prompt_tokens);
        strncpy(job->error_msg, "tokenise: buffer too small",
                sizeof(job->error_msg) - 1);
        return -1;
    }
    if (n_prompt == 0) {
        free(prompt_tokens);
        strncpy(job->error_msg, "tokenise: empty token list",
                sizeof(job->error_msg) - 1);
        return -1;
    }

    int max_new = job->max_tokens;
    if (n_prompt + max_new > n_ctx_max) {
        max_new = n_ctx_max - n_prompt;
        LOG_WARN("Job #%llu: max_tokens clamped to %d",
                 (unsigned long long)job->job_id, max_new);
    }

    LOG_DEBUG("Job #%llu: %d prompt tokens, generating up to %d",
              (unsigned long long)job->job_id, n_prompt, max_new);

    /* ── 2. Clear KV cache ──────────────────────────────────────────────── */
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, false);

    /* ── 3. Prefill ─────────────────────────────────────────────────────── */
    struct llama_batch batch = llama_batch_init(n_ctx_max, 0, 1);

    for (int i = 0; i < n_prompt; i++) {
        int want_logits = (i == n_prompt - 1) ? 1 : 0;
        batch_add_token(&batch, prompt_tokens[i], i, 0, want_logits);
    }

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        free(prompt_tokens);
        strncpy(job->error_msg, "llama_decode: prefill failed",
                sizeof(job->error_msg) - 1);
        return -1;
    }

    /* ── 4. Autoregressive decode ───────────────────────────────────────── */
    unsigned int rng_seed = (engine->config.seed < 0)
                            ? (unsigned int)time(NULL)
                            : (unsigned int)engine->config.seed;

    llama_token new_token;
    int         n_generated = 0;
    int         pos         = n_prompt;
    char        piece[256];

    while (n_generated < max_new) {
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

        if (job->temperature <= 0.0f)
            new_token = greedy_sample(logits, n_vocab);
        else
            new_token = temperature_sample(logits, n_vocab,
                                           job->temperature, &rng_seed);

        if (llama_vocab_is_eog(vocab, new_token)) {
            LOG_DEBUG("Job #%llu: EOG at step %d",
                      (unsigned long long)job->job_id, n_generated);
            break;
        }

        int piece_len = llama_token_to_piece(
            vocab, new_token,
            piece, (int32_t)sizeof(piece),
            0,      /* lstrip  */
            false   /* special */
        );
        if (piece_len < 0) piece_len = 0;

        if (job_append_output(job, piece, (size_t)piece_len) != 0) {
            LOG_ERROR("Job #%llu: output buffer exhausted",
                      (unsigned long long)job->job_id);
            break;
        }

        batch_clear(&batch);
        batch_add_token(&batch, new_token, pos, 0, 1);
        pos++;
        n_generated++;

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            free(prompt_tokens);
            strncpy(job->error_msg, "llama_decode: generation step failed",
                    sizeof(job->error_msg) - 1);
            return -1;
        }
    }

    llama_batch_free(batch);
    free(prompt_tokens);

    LOG_DEBUG("Job #%llu: generated %d tokens",
              (unsigned long long)job->job_id, n_generated);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Worker thread
 * ═══════════════════════════════════════════════════════════════════════════ */

static void *worker_thread_fn(void *arg)
{
    InferenceEngine *engine = (InferenceEngine *)arg;
    LOG_INFO("Worker thread started");

    while (1) {
        InferenceJob *job = queue_pop(engine->queue);
        if (!job) {
            LOG_INFO("Worker: queue drained, exiting");
            break;
        }

        job_mark_running(job);

        double t0      = now_ms();
        int    rc      = run_inference(engine, job);
        double elapsed = now_ms() - t0;
        double qwait   = job_queue_time_ms(job);  /* safe: job not yet freed */

        /* Update stats BEFORE broadcasting (job_mark_done/error wakes callers
         * who may immediately call job_destroy — reading job after that is UB) */
        pthread_mutex_lock(&engine->stats_mutex);
        engine->stats.total_exec_ms       += elapsed;
        engine->stats.total_queue_wait_ms += qwait;
        if (rc == 0) {
            engine->stats.jobs_completed++;
        } else {
            engine->stats.jobs_failed++;
        }
        pthread_mutex_unlock(&engine->stats_mutex);

        /* Now it is safe to broadcast — caller can destroy the job */
        if (rc == 0)
            job_mark_done(job);
        else
            job_mark_error(job, job->error_msg);
    }

    LOG_INFO("Worker thread exited");
    return NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

InferenceEngine *engine_create(const EngineConfig *config)
{
    if (!config) {
        LOG_ERROR("engine_create: config is NULL");
        return NULL;
    }
    if (config->model_path[0] == '\0') {
        LOG_ERROR("engine_create: model_path is empty");
        return NULL;
    }

    InferenceEngine *engine = calloc(1, sizeof(*engine));
    if (!engine) {
        LOG_ERROR("engine_create: out of memory");
        return NULL;
    }

    memcpy(&engine->config, config, sizeof(*config));

    if (engine->config.n_ctx              <= 0) engine->config.n_ctx              = 2048;
    if (engine->config.n_threads          <= 0) engine->config.n_threads          = 4;
    if (engine->config.default_max_tokens <= 0) engine->config.default_max_tokens = 256;
    if (engine->config.max_queue_size     == 0) engine->config.max_queue_size     = 256;

    pthread_mutex_init(&engine->stats_mutex, NULL);
    pthread_mutex_init(&engine->id_mutex,    NULL);

    llama_backend_init();

    /* ── Load model ─────────────────────────────────────────────────────── */
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = engine->config.n_gpu_layers;

    LOG_INFO("Loading model: %s", engine->config.model_path);
    engine->model = llama_model_load_from_file(engine->config.model_path, mparams);
    if (!engine->model) {
        LOG_ERROR("engine_create: failed to load model");
        goto fail_model;
    }
    LOG_INFO("Model loaded successfully");

    /* ── Create context ─────────────────────────────────────────────────── */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t)engine->config.n_ctx;
    cparams.n_threads       = (int32_t)engine->config.n_threads;
    cparams.n_threads_batch = (int32_t)engine->config.n_threads;

    engine->ctx = llama_init_from_model(engine->model, cparams);
    if (!engine->ctx) {
        LOG_ERROR("engine_create: failed to create context");
        goto fail_ctx;
    }
    LOG_INFO("Context created (n_ctx=%d, n_threads=%d)",
             engine->config.n_ctx, engine->config.n_threads);

    /* ── Job queue ──────────────────────────────────────────────────────── */
    engine->queue = queue_create((size_t)engine->config.max_queue_size);
    if (!engine->queue) {
        LOG_ERROR("engine_create: failed to create queue");
        goto fail_queue;
    }

    LOG_INFO("InferenceEngine ready");
    return engine;

fail_queue:
    llama_free(engine->ctx);
fail_ctx:
    llama_model_free(engine->model);
fail_model:
    llama_backend_free();
    pthread_mutex_destroy(&engine->stats_mutex);
    pthread_mutex_destroy(&engine->id_mutex);
    free(engine);
    return NULL;
}

int engine_start(InferenceEngine *engine)
{
    if (!engine) return -1;
    if (engine->running) {
        LOG_WARN("engine_start: already running");
        return 0;
    }
    engine->running = 1;
    int rc = pthread_create(&engine->worker_thread, NULL,
                            worker_thread_fn, engine);
    if (rc != 0) {
        engine->running = 0;
        LOG_ERROR("engine_start: pthread_create failed (rc=%d)", rc);
        return -1;
    }
    LOG_INFO("Engine started");
    return 0;
}

void engine_stop(InferenceEngine *engine)
{
    if (!engine || !engine->running) return;
    LOG_INFO("Engine stopping...");
    queue_shutdown(engine->queue);
    pthread_join(engine->worker_thread, NULL);
    engine->running = 0;
    LOG_INFO("Engine stopped");
}

void engine_destroy(InferenceEngine *engine)
{
    if (!engine) return;
    if (engine->running) engine_stop(engine);

    queue_destroy(engine->queue);
    llama_free(engine->ctx);
    llama_model_free(engine->model);
    llama_backend_free();
    pthread_mutex_destroy(&engine->stats_mutex);
    pthread_mutex_destroy(&engine->id_mutex);
    free(engine);
    LOG_INFO("Engine destroyed");
}

static uint64_t engine_next_id(InferenceEngine *engine)
{
    pthread_mutex_lock(&engine->id_mutex);
    uint64_t id = ++engine->next_job_id;
    pthread_mutex_unlock(&engine->id_mutex);
    return id;
}

InferenceJob *engine_submit_async(InferenceEngine *engine,
                                  const char *prompt,
                                  int max_tokens, float temperature)
{
    if (!engine || !prompt) return NULL;
    if (!engine->running) {
        LOG_ERROR("engine_submit_async: engine not running");
        return NULL;
    }

    int   tok = (max_tokens  > 0)  ? max_tokens  : engine->config.default_max_tokens;
    float tmp = (temperature >= 0) ? temperature : engine->config.default_temperature;

    InferenceJob *job = job_create(engine_next_id(engine), prompt, tok, tmp);
    if (!job) return NULL;

    pthread_mutex_lock(&engine->stats_mutex);
    engine->stats.jobs_submitted++;
    pthread_mutex_unlock(&engine->stats_mutex);

    if (queue_push(engine->queue, job) != 0) {
        LOG_ERROR("engine_submit_async: queue push failed");
        job_destroy(job);
        return NULL;
    }
    return job;
}

InferenceJob *engine_submit_sync(InferenceEngine *engine,
                                 const char *prompt,
                                 int max_tokens, float temperature)
{
    InferenceJob *job = engine_submit_async(engine, prompt,
                                            max_tokens, temperature);
    if (!job) return NULL;
    job_wait(job);
    return job;
}

void engine_get_stats(const InferenceEngine *engine, EngineStats *out)
{
    if (!engine || !out) return;
    pthread_mutex_lock((pthread_mutex_t *)&engine->stats_mutex);
    *out = engine->stats;
    pthread_mutex_unlock((pthread_mutex_t *)&engine->stats_mutex);
}

void engine_print_stats(const InferenceEngine *engine)
{
    if (!engine) return;
    EngineStats s;
    engine_get_stats(engine, &s);
    double avg_exec  = s.jobs_completed > 0 ? s.total_exec_ms / (double)s.jobs_completed : 0.0;
    double avg_queue = s.jobs_completed > 0 ? s.total_queue_wait_ms / (double)s.jobs_completed : 0.0;

    printf("\n╔══════════════════════════════════════╗\n");
    printf("║        Engine Statistics             ║\n");
    printf("╠══════════════════════════════════════╣\n");
    printf("║  Jobs submitted  : %-17llu ║\n", (unsigned long long)s.jobs_submitted);
    printf("║  Jobs completed  : %-17llu ║\n", (unsigned long long)s.jobs_completed);
    printf("║  Jobs failed     : %-17llu ║\n", (unsigned long long)s.jobs_failed);
    printf("║  Avg exec time   : %-14.1f ms ║\n", avg_exec);
    printf("║  Avg queue wait  : %-14.1f ms ║\n", avg_queue);
    printf("║  Total exec time : %-14.1f ms ║\n", s.total_exec_ms);
    printf("╚══════════════════════════════════════╝\n\n");
}
