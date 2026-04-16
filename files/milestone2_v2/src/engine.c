/*
 * engine.c — High-performance Parallel Inference Engine (Milestone 2)
 *
 * Optimisation 1: CPU AFFINITY
 *   pthread_setaffinity_np() pins each worker to a dedicated CPU core.
 *   Benefits:
 *     - Eliminates OS scheduler migrations (no cache invalidation between runs)
 *     - Each worker's L1/L2 cache stays warm for its own context data
 *     - Eliminates scheduling jitter in timing measurements
 *
 * Optimisation 2: LOCK-FREE STATS
 *   __atomic_fetch_add() updates counters without mutex contention.
 *   Workers never block each other recording results.
 *
 * Optimisation 3: TOKENS/SEC TRACKING
 *   Each job records n_tokens_generated for aggregate throughput reporting.
 *
 * Optimisation 4: SEPARATE CONTEXTS PER WORKER
 *   llama_decode() is not thread-safe within a single context.
 *   Each worker owns its own llama_context — calls truly run in parallel.
 */

#define _GNU_SOURCE          /* for pthread_setaffinity_np, CPU_SET */
#include "engine.h"
#include "logger.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <unistd.h>          /* sysconf, _SC_NPROCESSORS_ONLN */
#include <pthread.h>
#include <sched.h>           /* CPU_SET, CPU_ZERO */

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

static int tp_desc(const void *a, const void *b)
{
    float da = ((const TokenProb *)a)->prob;
    float db = ((const TokenProb *)b)->prob;
    return (da < db) - (da > db);
}

static llama_token temperature_sample(const float *logits, int n_vocab,
                                      float temp, unsigned int *seed)
{
    float *s = malloc((size_t)n_vocab * sizeof(float));
    if (!s) return greedy_sample(logits, n_vocab);

    float mx = -FLT_MAX;
    for (int i = 0; i < n_vocab; i++) {
        s[i] = logits[i] / temp;
        if (s[i] > mx) mx = s[i];
    }
    float sum = 0;
    for (int i = 0; i < n_vocab; i++) { s[i] = expf(s[i]-mx); sum += s[i]; }
    for (int i = 0; i < n_vocab; i++) s[i] /= sum;

    TokenProb *tp = malloc((size_t)n_vocab * sizeof(TokenProb));
    if (!tp) { free(s); return greedy_sample(logits, n_vocab); }
    for (int i = 0; i < n_vocab; i++) { tp[i].idx=i; tp[i].prob=s[i]; }
    qsort(tp, (size_t)n_vocab, sizeof(TokenProb), tp_desc);
    free(s);

    float cum = 0; int cut = n_vocab;
    for (int i = 0; i < n_vocab; i++) { cum += tp[i].prob; if (cum>=0.9f){cut=i+1;break;} }
    float norm = 0; for (int i=0;i<cut;i++) norm+=tp[i].prob;
    float r = ((float)(rand_r(seed)&0x7FFF))/(float)0x8000*norm;
    float acc=0; llama_token c=tp[0].idx;
    for (int i=0;i<cut;i++){acc+=tp[i].prob;if(r<=acc){c=tp[i].idx;break;}}
    free(tp);
    return c;
}

/* ── Batch helpers ──────────────────────────────────────────────────────── */

static void batch_clear(struct llama_batch *b) { b->n_tokens=0; }

static void batch_add(struct llama_batch *b, llama_token tok,
                      int pos, int seq, int want_logits)
{
    int n=b->n_tokens;
    b->token[n]=tok; b->pos[n]=pos;
    b->n_seq_id[n]=1; b->seq_id[n][0]=seq;
    b->logits[n]=(int8_t)want_logits;
    b->n_tokens++;
}

/* ── CPU affinity ───────────────────────────────────────────────────────── */

/*
 * pin_to_core() — Bind the calling thread to a specific CPU core.
 *
 * This is the key optimisation: each worker thread stays on its own core,
 * keeping its KV cache data warm in L1/L2.  Without pinning, the OS may
 * migrate threads between cores, invalidating cache lines and adding ~10%
 * overhead on matrix-vector multiplications inside llama_decode().
 */
static int pin_to_core(int core_id)
{
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET((size_t)core_id, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(),
                                    sizeof(cpu_set_t), &cpuset);
    return rc;
#else
    (void)core_id;
    return 0;   /* no-op on non-Linux */
#endif
}

/* ── Inference ──────────────────────────────────────────────────────────── */

static int run_inference(WorkerState *ws, InferenceJob *job)
{
    InferenceEngine          *eng   = (InferenceEngine *)ws->engine;
    struct llama_model       *model = eng->model;
    struct llama_context     *ctx   = ws->ctx;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_ctx_max             = (int)llama_n_ctx(ctx);
    const int n_vocab               = llama_vocab_n_tokens(vocab);

    /* 1. Tokenise */
    llama_token *toks = malloc((size_t)n_ctx_max * sizeof(llama_token));
    if (!toks) {
        strncpy(job->error_msg,"OOM tokenise",sizeof(job->error_msg)-1);
        return -1;
    }
    int np = llama_tokenize(vocab, job->prompt,
                            (int32_t)strlen(job->prompt),
                            toks, n_ctx_max, true, false);
    if (np <= 0) {
        free(toks);
        strncpy(job->error_msg, np<0?"prompt too long":"empty prompt",
                sizeof(job->error_msg)-1);
        return -1;
    }
    int max_new = job->max_tokens;
    if (np + max_new > n_ctx_max) max_new = n_ctx_max - np;

    LOG_DEBUG("[W%d/core%d] Job #%llu: %d prompt tokens, up to %d new",
              ws->worker_id, ws->cpu_core,
              (unsigned long long)job->job_id, np, max_new);

    /* 2. Clear KV cache (this worker's private context) */
    llama_memory_clear(llama_get_memory(ctx), false);

    /* 3. Prefill */
    struct llama_batch batch = llama_batch_init(n_ctx_max, 0, 1);
    for (int i = 0; i < np; i++)
        batch_add(&batch, toks[i], i, 0, i==np-1);

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch); free(toks);
        strncpy(job->error_msg,"prefill failed",sizeof(job->error_msg)-1);
        return -1;
    }

    /* 4. Autoregressive generation */
    unsigned int rng = (eng->config.seed < 0)
                       ? (unsigned int)time(NULL) ^ (unsigned int)ws->worker_id
                       : (unsigned int)(eng->config.seed + ws->worker_id);

    int pos=np, n_gen=0;
    char piece[256];

    while (n_gen < max_new) {
        float *logits = llama_get_logits_ith(ctx, batch.n_tokens-1);
        llama_token tok = (job->temperature <= 0.0f)
                          ? greedy_sample(logits, n_vocab)
                          : temperature_sample(logits, n_vocab,
                                               job->temperature, &rng);

        if (llama_vocab_is_eog(vocab, tok)) break;

        int pl = llama_token_to_piece(vocab, tok, piece,
                                      (int32_t)sizeof(piece), 0, false);
        if (pl < 0) pl = 0;
        if (job_append_output(job, piece, (size_t)pl) != 0) break;

        batch_clear(&batch);
        batch_add(&batch, tok, pos++, 0, 1);
        n_gen++;

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch); free(toks);
            strncpy(job->error_msg,"decode failed",sizeof(job->error_msg)-1);
            return -1;
        }
    }

    job->n_tokens_generated = n_gen;
    llama_batch_free(batch);
    free(toks);
    return 0;
}

/* ── Worker thread ──────────────────────────────────────────────────────── */

static void *worker_thread_fn(void *arg)
{
    WorkerState     *ws  = (WorkerState *)arg;
    InferenceEngine *eng = (InferenceEngine *)ws->engine;

    /* ── OPTIMISATION 1: Pin this thread to its dedicated CPU core ── */
    if (eng->config.use_cpu_affinity && ws->cpu_core >= 0) {
        if (pin_to_core(ws->cpu_core) == 0) {
            LOG_INFO("[W%d] pinned to CPU core %d",
                     ws->worker_id, ws->cpu_core);
        } else {
            LOG_WARN("[W%d] CPU affinity failed (continuing without pin)",
                     ws->worker_id);
        }
    }

    LOG_INFO("[W%d/core%d] Worker thread started",
             ws->worker_id, ws->cpu_core);

    while (1) {
        InferenceJob *job = queue_pop(eng->queue);
        if (!job) { LOG_INFO("[W%d] exiting", ws->worker_id); break; }

        job_mark_running(job);

        double t0  = now_ms();
        int    rc  = run_inference(ws, job);
        double ms  = now_ms() - t0;

        /* ── OPTIMISATION 2: Lock-free stat updates ── */
        uint64_t exec_ns = (uint64_t)(ms * 1e6);
        uint64_t wait_ns = (uint64_t)(job_queue_time_ms(job) * 1e6);

        __atomic_fetch_add(&eng->stats.total_exec_ns,  exec_ns, __ATOMIC_RELAXED);
        __atomic_fetch_add(&eng->stats.total_queue_ns, wait_ns, __ATOMIC_RELAXED);

        if (rc == 0) {
            __atomic_fetch_add(&eng->stats.jobs_completed,  1,
                               __ATOMIC_RELAXED);
            __atomic_fetch_add(&eng->stats.total_tokens,
                               (uint64_t)job->n_tokens_generated,
                               __ATOMIC_RELAXED);
            job_mark_done(job);
        } else {
            __atomic_fetch_add(&eng->stats.jobs_failed, 1,
                               __ATOMIC_RELAXED);
            job_mark_error(job, job->error_msg);
        }
    }
    return NULL;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

InferenceEngine *engine_create(const EngineConfig *config)
{
    if (!config) return NULL;

    InferenceEngine *e = calloc(1, sizeof(*e));
    if (!e) return NULL;
    memcpy(&e->config, config, sizeof(*config));

    if (e->config.n_ctx              <= 0) e->config.n_ctx              = 2048;
    if (e->config.n_threads          <= 0) e->config.n_threads          = 1;
    if (e->config.n_workers          <= 0) e->config.n_workers          = 1;
    if (e->config.default_max_tokens <= 0) e->config.default_max_tokens = 256;
    if (e->config.max_queue_size     == 0) e->config.max_queue_size     = 512;

    pthread_mutex_init(&e->stats_mutex, NULL);
    pthread_mutex_init(&e->id_mutex,    NULL);

    /* ── Model ─────────────────────────────────────────────────────────── */
    if (config->preloaded_model) {
        e->model       = config->preloaded_model;
        e->model_owned = 0;
        LOG_INFO("Using preloaded model (shared across workers)");
    } else {
        llama_backend_init();
        struct llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = e->config.n_gpu_layers;
        LOG_INFO("Loading model: %s", e->config.model_path);
        e->model = llama_model_load_from_file(e->config.model_path, mp);
        if (!e->model) { LOG_ERROR("Failed to load model"); goto fail_model; }
        e->model_owned = 1;
    }

    /* ── Worker arrays ─────────────────────────────────────────────────── */
    int nw = e->config.n_workers;
    e->worker_states  = calloc((size_t)nw, sizeof(WorkerState));
    e->worker_threads = calloc((size_t)nw, sizeof(pthread_t));
    if (!e->worker_states || !e->worker_threads) goto fail_workers;

    /* ── One context per worker ────────────────────────────────────────── */
    int n_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);

    for (int i = 0; i < nw; i++) {
        struct llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = (uint32_t)e->config.n_ctx;
        cp.n_threads       = (int32_t)e->config.n_threads;
        cp.n_threads_batch = (int32_t)e->config.n_threads;

        e->worker_states[i].worker_id = i;
        e->worker_states[i].engine    = e;
        /* Assign core round-robin, staying within available cores */
        e->worker_states[i].cpu_core  =
            e->config.use_cpu_affinity ? (i % n_cpus) : -1;

        e->worker_states[i].ctx = llama_init_from_model(e->model, cp);
        if (!e->worker_states[i].ctx) {
            LOG_ERROR("Failed to create context for worker %d", i);
            for (int j=0;j<i;j++) llama_free(e->worker_states[j].ctx);
            goto fail_workers;
        }
        LOG_INFO("Worker %d context ready (will pin to core %d)",
                 i, e->worker_states[i].cpu_core);
    }

    e->queue = queue_create((size_t)e->config.max_queue_size);
    if (!e->queue) {
        for (int i=0;i<nw;i++) llama_free(e->worker_states[i].ctx);
        goto fail_workers;
    }

    LOG_INFO("Engine ready (%d workers, affinity=%s)",
             nw, e->config.use_cpu_affinity ? "ON" : "OFF");
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
            for (int j=0;j<i;j++) pthread_join(e->worker_threads[j], NULL);
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
    for (int i=0;i<e->config.n_workers;i++)
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
    for (int i=0;i<e->config.n_workers;i++) llama_free(e->worker_states[i].ctx);
    free(e->worker_states);
    free(e->worker_threads);
    if (e->model_owned) { llama_model_free(e->model); llama_backend_free(); }
    pthread_mutex_destroy(&e->stats_mutex);
    pthread_mutex_destroy(&e->id_mutex);
    free(e);
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

    __atomic_fetch_add(&e->stats.jobs_submitted, 1, __ATOMIC_RELAXED);

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
    /* Atomic loads — consistent snapshot without mutex */
    out->jobs_submitted = __atomic_load_n(&e->stats.jobs_submitted, __ATOMIC_SEQ_CST);
    out->jobs_completed = __atomic_load_n(&e->stats.jobs_completed, __ATOMIC_SEQ_CST);
    out->jobs_failed    = __atomic_load_n(&e->stats.jobs_failed,    __ATOMIC_SEQ_CST);
    out->total_exec_ns  = __atomic_load_n(&e->stats.total_exec_ns,  __ATOMIC_SEQ_CST);
    out->total_queue_ns = __atomic_load_n(&e->stats.total_queue_ns, __ATOMIC_SEQ_CST);
    out->total_tokens   = __atomic_load_n(&e->stats.total_tokens,   __ATOMIC_SEQ_CST);
    pthread_mutex_lock((pthread_mutex_t *)&e->stats_mutex);
    out->wall_time_ms = e->stats.wall_time_ms;
    pthread_mutex_unlock((pthread_mutex_t *)&e->stats_mutex);
}

void engine_print_stats(const InferenceEngine *e)
{
    if (!e) return;
    EngineStats s; engine_get_stats(e, &s);

    double avg_exec  = s.jobs_completed > 0 ? (s.total_exec_ns/1e6) / s.jobs_completed : 0;
    double avg_queue = s.jobs_completed > 0 ? (s.total_queue_ns/1e6) / s.jobs_completed : 0;
    double tput_jobs = s.wall_time_ms > 0 ? s.jobs_completed / (s.wall_time_ms/1000.0) : 0;
    double tput_toks = s.wall_time_ms > 0 ? s.total_tokens   / (s.wall_time_ms/1000.0) : 0;

    printf("  Workers=%d | Jobs=%llu | Tokens=%llu\n",
           e->config.n_workers,
           (unsigned long long)s.jobs_completed,
           (unsigned long long)s.total_tokens);
    printf("  Wall=%.1f ms | AvgExec=%.1f ms | AvgQueue=%.1f ms\n",
           s.wall_time_ms, avg_exec, avg_queue);
    printf("  Throughput: %.3f jobs/s | %.1f tokens/s\n\n",
           tput_jobs, tput_toks);
}
