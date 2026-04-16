#include "job.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void ts_now(struct timespec *ts) { clock_gettime(CLOCK_MONOTONIC, ts); }

static double ts_diff_ms(const struct timespec *a, const struct timespec *b)
{
    return (b->tv_sec - a->tv_sec) * 1000.0 + (b->tv_nsec - a->tv_nsec) / 1e6;
}

InferenceJob *job_create(uint64_t id, const char *prompt,
                         int max_tokens, float temperature)
{
    if (!prompt) return NULL;
    InferenceJob *j = calloc(1, sizeof(*j));
    if (!j) return NULL;

    j->job_id      = id;
    j->max_tokens  = max_tokens  > 0  ? max_tokens  : 256;
    j->temperature = temperature >= 0 ? temperature : 0.0f;
    j->status      = JOB_STATUS_PENDING;

    j->prompt = strdup(prompt);
    if (!j->prompt) { free(j); return NULL; }

    j->output_cap = 64;
    j->output     = malloc(j->output_cap);
    if (!j->output) { free(j->prompt); free(j); return NULL; }
    j->output[0]  = '\0';

    pthread_mutex_init(&j->mutex, NULL);
    pthread_cond_init(&j->cond,   NULL);
    ts_now(&j->submit_time);
    return j;
}

void job_destroy(InferenceJob *j)
{
    if (!j) return;
    pthread_mutex_destroy(&j->mutex);
    pthread_cond_destroy(&j->cond);
    free(j->prompt);
    free(j->output);
    free(j);
}

int job_append_output(InferenceJob *j, const char *frag, size_t len)
{
    size_t need = j->output_len + len + 1;
    if (need > j->output_cap) {
        size_t nc = j->output_cap * 2;
        while (nc < need) nc *= 2;
        char *tmp = realloc(j->output, nc);
        if (!tmp) return -1;
        j->output     = tmp;
        j->output_cap = nc;
    }
    memcpy(j->output + j->output_len, frag, len);
    j->output_len += len;
    j->output[j->output_len] = '\0';
    return 0;
}

void job_mark_running(InferenceJob *j)
{
    pthread_mutex_lock(&j->mutex);
    j->status = JOB_STATUS_RUNNING;
    ts_now(&j->start_time);
    pthread_mutex_unlock(&j->mutex);
}

void job_mark_done(InferenceJob *j)
{
    pthread_mutex_lock(&j->mutex);
    j->status = JOB_STATUS_DONE;
    ts_now(&j->end_time);
    pthread_cond_broadcast(&j->cond);
    pthread_mutex_unlock(&j->mutex);
}

void job_mark_error(InferenceJob *j, const char *reason)
{
    pthread_mutex_lock(&j->mutex);
    j->status = JOB_STATUS_ERROR;
    ts_now(&j->end_time);
    if (reason) {
        strncpy(j->error_msg, reason, sizeof(j->error_msg) - 1);
        j->error_msg[sizeof(j->error_msg) - 1] = '\0';
    }
    pthread_cond_broadcast(&j->cond);
    pthread_mutex_unlock(&j->mutex);
}

void job_wait(InferenceJob *j)
{
    pthread_mutex_lock(&j->mutex);
    while (j->status == JOB_STATUS_PENDING || j->status == JOB_STATUS_RUNNING)
        pthread_cond_wait(&j->cond, &j->mutex);
    pthread_mutex_unlock(&j->mutex);
}

double job_queue_time_ms(const InferenceJob *j)
{
    if (!j->start_time.tv_sec) return 0.0;
    return ts_diff_ms(&j->submit_time, &j->start_time);
}

double job_exec_time_ms(const InferenceJob *j)
{
    if (!j->end_time.tv_sec) return 0.0;
    return ts_diff_ms(&j->start_time, &j->end_time);
}

double job_total_time_ms(const InferenceJob *j)
{
    if (!j->end_time.tv_sec) return 0.0;
    return ts_diff_ms(&j->submit_time, &j->end_time);
}

double job_tokens_per_sec(const InferenceJob *j)
{
    double ms = job_exec_time_ms(j);
    if (ms <= 0.0 || j->n_tokens_generated <= 0) return 0.0;
    return (double)j->n_tokens_generated / (ms / 1000.0);
}

const char *job_status_str(JobStatus s)
{
    switch (s) {
        case JOB_STATUS_PENDING: return "PENDING";
        case JOB_STATUS_RUNNING: return "RUNNING";
        case JOB_STATUS_DONE:    return "DONE";
        case JOB_STATUS_ERROR:   return "ERROR";
        default:                 return "UNKNOWN";
    }
}
