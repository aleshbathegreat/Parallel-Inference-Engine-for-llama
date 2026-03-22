/*
 * job.c — Inference job lifecycle management
 */

#include "job.h"
#include "logger.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Internal helpers ───────────────────────────────────────────────────── */

static void ts_now(struct timespec *ts)
{
    clock_gettime(CLOCK_MONOTONIC, ts);
}

/* Elapsed milliseconds between two CLOCK_MONOTONIC timestamps. */
static double ts_diff_ms(const struct timespec *start,
                          const struct timespec *end)
{
    double sec  = (double)(end->tv_sec  - start->tv_sec);
    double nsec = (double)(end->tv_nsec - start->tv_nsec);
    return sec * 1000.0 + nsec / 1.0e6;
}

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

InferenceJob *job_create(uint64_t job_id, const char *prompt,
                         int max_tokens, float temperature)
{
    if (!prompt) {
        LOG_ERROR("job_create: prompt must not be NULL");
        return NULL;
    }

    InferenceJob *job = calloc(1, sizeof(*job));
    if (!job) {
        LOG_ERROR("job_create: out of memory");
        return NULL;
    }

    job->job_id      = job_id;
    job->max_tokens  = (max_tokens > 0)    ? max_tokens  : 256;
    job->temperature = (temperature >= 0)  ? temperature : 0.0f;
    job->status      = JOB_STATUS_PENDING;

    /* Copy prompt */
    job->prompt = strdup(prompt);
    if (!job->prompt) {
        LOG_ERROR("job_create: strdup failed for prompt");
        free(job);
        return NULL;
    }

    /* Allocate initial output buffer (64 bytes; grows as needed) */
    job->output_cap = 64;
    job->output     = malloc(job->output_cap);
    if (!job->output) {
        LOG_ERROR("job_create: failed to allocate output buffer");
        free(job->prompt);
        free(job);
        return NULL;
    }
    job->output[0]  = '\0';
    job->output_len = 0;

    /* Synchronisation primitives */
    pthread_mutex_init(&job->mutex, NULL);
    pthread_cond_init(&job->cond, NULL);

    /* Record submission time */
    ts_now(&job->submit_time);

    LOG_DEBUG("Job #%llu created | max_tokens=%d temp=%.2f prompt=\"%.40s%s\"",
              (unsigned long long)job_id,
              job->max_tokens,
              (double)job->temperature,
              prompt,
              strlen(prompt) > 40 ? "..." : "");

    return job;
}

void job_destroy(InferenceJob *job)
{
    if (!job) return;

    pthread_mutex_destroy(&job->mutex);
    pthread_cond_destroy(&job->cond);
    free(job->prompt);
    free(job->output);
    free(job);
}

/* ── Output helpers ─────────────────────────────────────────────────────── */

int job_append_output(InferenceJob *job, const char *fragment, size_t len)
{
    if (!job || !fragment || len == 0) return 0;

    /* Grow buffer if needed (+1 for NUL terminator) */
    size_t required = job->output_len + len + 1;
    if (required > job->output_cap) {
        size_t new_cap = job->output_cap * 2;
        while (new_cap < required) new_cap *= 2;

        char *tmp = realloc(job->output, new_cap);
        if (!tmp) {
            LOG_ERROR("Job #%llu: output buffer realloc failed",
                      (unsigned long long)job->job_id);
            return -1;
        }
        job->output     = tmp;
        job->output_cap = new_cap;
    }

    memcpy(job->output + job->output_len, fragment, len);
    job->output_len        += len;
    job->output[job->output_len] = '\0';
    return 0;
}

/* ── Status transitions ─────────────────────────────────────────────────── */

void job_mark_running(InferenceJob *job)
{
    pthread_mutex_lock(&job->mutex);
    job->status = JOB_STATUS_RUNNING;
    ts_now(&job->start_time);
    pthread_mutex_unlock(&job->mutex);

    LOG_DEBUG("Job #%llu → RUNNING  (queue_wait=%.1f ms)",
              (unsigned long long)job->job_id,
              job_queue_time_ms(job));
}

void job_mark_done(InferenceJob *job)
{
    pthread_mutex_lock(&job->mutex);
    job->status = JOB_STATUS_DONE;
    ts_now(&job->end_time);
    pthread_cond_broadcast(&job->cond);
    pthread_mutex_unlock(&job->mutex);

    LOG_DEBUG("Job #%llu → DONE     (exec=%.1f ms, output_len=%zu)",
              (unsigned long long)job->job_id,
              job_exec_time_ms(job),
              job->output_len);
}

void job_mark_error(InferenceJob *job, const char *reason)
{
    pthread_mutex_lock(&job->mutex);
    job->status = JOB_STATUS_ERROR;
    ts_now(&job->end_time);
    if (reason) {
        strncpy(job->error_msg, reason, sizeof(job->error_msg) - 1);
        job->error_msg[sizeof(job->error_msg) - 1] = '\0';
    }
    pthread_cond_broadcast(&job->cond);
    pthread_mutex_unlock(&job->mutex);

    LOG_ERROR("Job #%llu → ERROR    (%s)",
              (unsigned long long)job->job_id,
              job->error_msg);
}

/* ── Waiting ────────────────────────────────────────────────────────────── */

void job_wait(InferenceJob *job)
{
    pthread_mutex_lock(&job->mutex);
    while (job->status == JOB_STATUS_PENDING ||
           job->status == JOB_STATUS_RUNNING) {
        pthread_cond_wait(&job->cond, &job->mutex);
    }
    pthread_mutex_unlock(&job->mutex);
}

/* ── Timing ─────────────────────────────────────────────────────────────── */

double job_queue_time_ms(const InferenceJob *job)
{
    if (job->start_time.tv_sec == 0) return 0.0;   /* not started yet */
    return ts_diff_ms(&job->submit_time, &job->start_time);
}

double job_exec_time_ms(const InferenceJob *job)
{
    if (job->end_time.tv_sec == 0) return 0.0;      /* not finished yet */
    return ts_diff_ms(&job->start_time, &job->end_time);
}

double job_total_time_ms(const InferenceJob *job)
{
    if (job->end_time.tv_sec == 0) return 0.0;
    return ts_diff_ms(&job->submit_time, &job->end_time);
}

/* ── Utility ────────────────────────────────────────────────────────────── */

const char *job_status_str(JobStatus status)
{
    switch (status) {
        case JOB_STATUS_PENDING: return "PENDING";
        case JOB_STATUS_RUNNING: return "RUNNING";
        case JOB_STATUS_DONE:    return "DONE";
        case JOB_STATUS_ERROR:   return "ERROR";
        default:                 return "UNKNOWN";
    }
}
