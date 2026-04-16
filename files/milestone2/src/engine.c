/*
 * engine.c — Parallel Inference Engine (Milestone 2)
 *
 * Key changes from Milestone 1:
 * - N worker threads, each with their own llama_context.
 * - config.preloaded_model lets callers share one model across multiple
 *   engine instances (avoids reloading 636 MB between benchmark runs).
 * - engine.model_owned tracks whether we loaded the model (and must free it).
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

/* ── Timing ─────────────────────────────────────────────────────────────── */

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1.0e6;
}

static double ts_to_ms(const struct timespec *ts)
{
    return (double)ts->tv_sec * 1000.0 + (double)ts->tv_nsec / 1.0e6;
}

/* ── Sampling ───────────────────────────────────────────────────────────── */

static llama_token greedy_sample(const float *logits, int n_vocab)
{
    llama_token best = 0;
    float best_l = -FLT_MAX;
    for (int i = 0; i < n_vocab; i++)
        if (logits[i] > best_l) { best_l = logits[i]; best = i; }
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
    for (int i = 0; i < n_vocab; i++) { scaled[i] = expf(scaled[i]-max_l); sum += scaled[i]; }
    for (int i = 0; i < n_vocab; i++) scaled[i] /= sum;

    TokenProb *tp = malloc((size_t)n_vocab * sizeof(TokenProb));
    if (!tp) { free(scaled); return greedy_sample(logits, n_vocab); }
    for (int i = 0; i < n_vocab; i++) { tp[i].idx = i; tp[i].prob = scaled[i]; }
    qsort(tp, (size_t)n_vocab, sizeof(TokenProb), tp_cmp_desc);
    free(scaled);

    float cumsum = 0.0f; int cutoff = n_vocab;
    for (int i = 0; i < n_vocab; i++) { cumsum += tp[i].prob; if (cumsum >= 0.9f) { cutoff = i+1; break; } }
    float norm = 0.0f;
    for (int i = 0; i < cutoff; i++) norm += tp[i].prob;
    float r = ((float)(rand_r(seed) & 0x7FFF)) / (float)0x8000 * norm;
    float acc = 0.0f; llama_token chosen = tp[0].idx;
    for (int i = 0; i < cutoff; i++) { acc += tp[i].prob; if (r <= acc) { chosen = tp[i].idx; break; } }
    free(tp);
    return chosen;
}

/* ── Batch helpers ──────────────────────────────────────────────────────── */

static void batch_clear(struct llama_batch *b) { b->n_tokens = 0; }

static void batch_add(struct llama_batch *b, llama_token tok,
                      int pos, int seq_id, int want_logits)
{
    int n = b->n_tokens;
    b->token[n]=tok; b->pos[n]=pos;
    b->n_seq_id[n]=1; b->seq_id[n][0]=seq_id;
    b->logits[n]=(int8_t)want_logits;
    b->n_tokens++;
}

/* ── Inference ──────────────────────────────────────────────────────────── */

static int run_inference(WorkerState *ws, InferenceJob *job)
{
    InferenceEngine          *engine = (InferenceEngine *)ws->engine;
    struct llama_model       *model  = engine->model;
    struct llama_context     *ctx    = ws->ctx;
    const struct llama_vocab *vocab  = llama_model_get_vocab(model);
    const int n_ctx_max              = (int)llama_n_ctx(ctx);
    const int n_vocab                = llama_vocab_n_tokens(vocab);

    /* 1. Tokenise */
    llama_token *tokens = malloc((size_t)n_ctx_max * sizeof(llama_token));
    if (!tokens) {
        strncpy(job->error_msg, "out of memory", sizeof(job->error_msg)-1);
        return -1;
    }

    int n_prompt = llama_tokenize(vocab, job->prompt,
                                  (int32_t)strlen(job->prompt),
                                  tokens, n_ctx_max, true, false);
    if (n_prompt <= 0) {
        free(tokens);
        strncpy(job->error_msg, n_prompt < 0 ? "tokenise: too long" : "tokenise: empty",
                sizeof(job->error_msg)-1);
        return -1;
    }

    int max_new = job->max_tokens;
    if (n_prompt + max_new > n_ctx_max) max_new = n_ctx_max - n_prompt;

    LOG_DEBUG("[W%d] Job #%llu: %d prompt tokens, up to %d new",
              ws->worker_id, (unsigned long long)job->job_id, n_prompt, max_new);

    /* 2. Clear this worker's KV cache */
    llama_memory_clear(llama_get_memory(ctx), false);

    /* 3. Prefill */
    struct llama_batch batch = llama_batch_init(n_ctx_max, 0, 1);
    for (int i = 0; i < n_prompt; i++)
        batch_add(&batch, tokens[i], i, 0, i == n_prompt-1);

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch); free(tokens);
        strncpy(job->error_msg, "prefill failed", sizeof(job->error_msg)-1);
        return -1;
    }

    /* 4. Generate */
    unsigned int rng = (engine->config.seed < 0)
                       ? (unsigned int)time(NULL) ^ (unsigned int)ws->worker_id
                       : (unsigned int)(engine->config.seed + ws->worker_id);

    int pos = n_prompt, n_gen = 0;
    char piece[256];

    while (n_gen < max_new) {
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        llama_token tok = (job->temperature <= 0.0f)
                          ? greedy_sample(logits, n_vocab)
                          : temperature_sample(logits, n_vocab, job->temperature, &rng);

        if (llama_vocab_is_eog(vocab, tok)) break;

        int plen = llama_token_to_piece(vocab, tok, piece, (int32_t)sizeof(piece), 0, false);
        if (plen < 0) plen = 0;
        if (job_append_output(job, piece, (size_t)plen) != 0) break;

        batch_clear(&batch);
        batch_add(&batch, tok, pos++, 0, 1);
        n_gen++;

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch); free(tokens);
            strncpy(job->error_msg, "decode failed", sizeof(job->error_msg)-1);
            return -1;
        }
    }

    llama_batch_free(batch);
    free(tokens);
    LOG_DEBUG("[W%d] Job #%llu: generated %d tokens",
              ws->worker_id, (unsigned long long)job->job_id, n_gen);
    return 0;
}

/* ── Worker thread ──────────────────────────────────────────────────────── */

static void *worker_thread_fn(void *arg)
{
    WorkerState     *ws     = (WorkerState *)arg;
    InferenceEngine *engine = (InferenceEngine *)ws->engine;

    LOG_INFO("[W%d] started", ws->worker_id);

    while (1) {
        InferenceJob *job = queue_pop(engine->queue);
        if (!job) { LOG_INFO("[W%d] exiting", ws->worker_id); break; }

        job_mark_running(job);

        double t0      = now_ms();
        int    rc      = run_inference(ws, job);
        double elapsed = now_ms() - t0;
        double qwait   = job_queue_time_ms(job);

        pthread_mutex_lock(&engine->stats_mutex);
        engine->stats.total_exec_ms       += elapsed;
        engine->stats.total_queue_wait_ms += qwait;
        if (rc == 0) engine->stats.jobs_completed++;
        else         engine->stats.jobs_failed++;
        pthread_mutex_unlock(&engine->stats_mutex);

        if (rc == 0) job_mark_done(job);
        else         job_mark_error(job, job->error_msg);
    }
    return NULL;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

InferenceEngine *engine_create(const EngineConfig *config)
{
    if (!config) { LOG_ERROR("engine_create: NULL config"); return NULL; }

    InferenceEngine *e = calloc(1, sizeof(*e));
    if (!e) { LOG_ERROR("engine_create: OOM"); return NULL; }

    memcpy(&e->config, config, sizeof(*config));
    if (e->config.n_ctx              <= 0) e->config.n_ctx              = 2048;
    if (e->config.n_threads          <= 0) e->config.n_threads          = 4;
    if (e->config.n_workers          <= 0) e->config.n_workers          = 1;
    if (e->config.default_max_tokens <= 0) e->config.default_max_tokens = 256;
    if (e->config.max_queue_size     == 0) e->config.max_queue_size     = 512;

    pthread_mutex_init(&e->stats_mutex, NULL);
    pthread_mutex_init(&e->id_mutex,    NULL);

    /* ── Model: use preloaded or load from file ─────────────────────────── */
    if (config->preloaded_model) {
        /* Caller owns this model — we must NOT free it in engine_destroy */
        e->model       = config->preloaded_model;
        e->model_owned = 0;
        LOG_INFO("Using preloaded model (shared)");
    } else {
        llama_backend_init();
        struct llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = e->config.n_gpu_layers;
        LOG_INFO("Loading model: %s", e->config.model_path);
        e->model = llama_model_load_from_file(e->config.model_path, mp);
        if (!e->model) {
            LOG_ERROR("Failed to load model");
            goto fail_model;
        }
        e->model_owned = 1;
        LOG_INFO("Model loaded");
    }

    /* ── Worker arrays ──────────────────────────────────────────────────── */
    int nw = e->config.n_workers;
    e->worker_states  = calloc((size_t)nw, sizeof(WorkerState));
    e->worker_threads = calloc((size_t)nw, sizeof(pthread_t));
    if (!e->worker_states || !e->worker_threads) {
        LOG_ERROR("Failed to alloc worker arrays");
        goto fail_workers;
    }

    /* ── One context per worker ─────────────────────────────────────────── */
    for (int i = 0; i < nw; i++) {
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t)e->config.n_ctx;
        cp.n_threads       = (int32_t)e->config.n_threads;
        cp.n_threads_batch = (int32_t)e->config.n_threads;

        e->worker_states[i].worker_id = i;
        e->worker_states[i].engine    = e;
        e->worker_states[i].ctx = llama_init_from_model(e->model, cp);

        if (!e->worker_states[i].ctx) {
            LOG_ERROR("Failed to create context for worker %d", i);
            for (int j = 0; j < i; j++) llama_free(e->worker_states[j].ctx);
            goto fail_workers;
        }
        LOG_INFO("Worker %d context ready", i);
    }

    /* ── Queue ──────────────────────────────────────────────────────────── */
    e->queue = queue_create((size_t)e->config.max_queue_size);
    if (!e->queue) {
        LOG_ERROR("Failed to create queue");
        for (int i = 0; i < nw; i++) llama_free(e->worker_states[i].ctx);
        goto fail_workers;
    }

    LOG_INFO("Engine ready (%d workers)", nw);
    return e;

fail_workers:
    free(e->worker_states);
    free(e->worker_threads);
    if (e->model_owned) { llama_model_free(e->model); llama_backend_free(); }
fail_model:
    pthread_mutex_destroy(&e->stats_mutex);
    pthread_mutex_destroy(&e->id_mutex);
    free(e);
    return NULL;
}

int engine_start(InferenceEngine *e)
{
    if (!e || e->running) return e ? 0 : -1;
    e->running = 1;
    clock_gettime(CLOCK_MONOTONIC, &e->start_time);

    for (int i = 0; i < e->config.n_workers; i++) {
        if (pthread_create(&e->worker_threads[i], NULL,
                           worker_thread_fn, &e->worker_states[i]) != 0) {
            LOG_ERROR("pthread_create failed for worker %d", i);
            queue_shutdown(e->queue);
            for (int j = 0; j < i; j++) pthread_join(e->worker_threads[j], NULL);
            e->running = 0;
            return -1;
        }
    }
    LOG_INFO("Engine started (%d workers)", e->config.n_workers);
    return 0;
}

void engine_stop(InferenceEngine *e)
{
    if (!e || !e->running) return;
    queue_shutdown(e->queue);
    for (int i = 0; i < e->config.n_workers; i++)
        pthread_join(e->worker_threads[i], NULL);

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    pthread_mutex_lock(&e->stats_mutex);
    e->stats.wall_time_ms = ts_to_ms(&now) - ts_to_ms(&e->start_time);
    pthread_mutex_unlock(&e->stats_mutex);

    e->running = 0;
    LOG_INFO("Engine stopped");
}

void engine_destroy(InferenceEngine *e)
{
    if (!e) return;
    if (e->running) engine_stop(e);

    queue_destroy(e->queue);
    for (int i = 0; i < e->config.n_workers; i++)
        llama_free(e->worker_states[i].ctx);
    free(e->worker_states);
    free(e->worker_threads);

    /* Only free the model if we loaded it */
    if (e->model_owned) {
        llama_model_free(e->model);
        llama_backend_free();
    }

    pthread_mutex_destroy(&e->stats_mutex);
    pthread_mutex_destroy(&e->id_mutex);
    free(e);
    LOG_INFO("Engine destroyed");
}

static uint64_t next_id(InferenceEngine *e)
{
    pthread_mutex_lock(&e->id_mutex);
    uint64_t id = ++e->next_job_id;
    pthread_mutex_unlock(&e->id_mutex);
    return id;
}

InferenceJob *engine_submit_async(InferenceEngine *e, const char *prompt,
                                  int max_tokens, float temperature)
{
    if (!e || !prompt || !e->running) return NULL;
    int   tok = max_tokens  > 0  ? max_tokens  : e->config.default_max_tokens;
    float tmp = temperature >= 0 ? temperature : e->config.default_temperature;

    InferenceJob *job = job_create(next_id(e), prompt, tok, tmp);
    if (!job) return NULL;

    pthread_mutex_lock(&e->stats_mutex);
    e->stats.jobs_submitted++;
    pthread_mutex_unlock(&e->stats_mutex);

    if (queue_push(e->queue, job) != 0) { job_destroy(job); return NULL; }
    return job;
}

InferenceJob *engine_submit_sync(InferenceEngine *e, const char *prompt,
                                 int max_tokens, float temperature)
{
    InferenceJob *job = engine_submit_async(e, prompt, max_tokens, temperature);
    if (!job) return NULL;
    job_wait(job);
    return job;
}

void engine_get_stats(const InferenceEngine *e, EngineStats *out)
{
    if (!e || !out) return;
    pthread_mutex_lock((pthread_mutex_t *)&e->stats_mutex);
    *out = e->stats;
    pthread_mutex_unlock((pthread_mutex_t *)&e->stats_mutex);
}

void engine_print_stats(const InferenceEngine *e)
{
    if (!e) return;
    EngineStats s; engine_get_stats(e, &s);
    double avg_exec  = s.jobs_completed > 0 ? s.total_exec_ms / s.jobs_completed : 0;
    double avg_queue = s.jobs_completed > 0 ? s.total_queue_wait_ms / s.jobs_completed : 0;
    double tput      = s.wall_time_ms   > 0 ? s.jobs_completed / (s.wall_time_ms/1000.0) : 0;

    printf("\n╔══════════════════════════════════════════╗\n");
    printf("║  Engine Stats (workers = %-2d)             ║\n", e->config.n_workers);
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Jobs submitted   : %-20llu ║\n", (unsigned long long)s.jobs_submitted);
    printf("║  Jobs completed   : %-20llu ║\n", (unsigned long long)s.jobs_completed);
    printf("║  Jobs failed      : %-20llu ║\n", (unsigned long long)s.jobs_failed);
    printf("║  Avg exec time    : %-17.1f ms ║\n", avg_exec);
    printf("║  Avg queue wait   : %-17.1f ms ║\n", avg_queue);
    printf("║  Total exec time  : %-17.1f ms ║\n", s.total_exec_ms);
    printf("║  Wall-clock time  : %-17.1f ms ║\n", s.wall_time_ms);
    printf("║  Throughput       : %-14.3f jobs/s ║\n", tput);
    printf("╚══════════════════════════════════════════╝\n\n");
}
