#ifndef JOB_H
#define JOB_H

/*
 * job.h — Inference job representation
 *
 * Extended for Milestone 2:
 *   - n_tokens_generated: tracks output token count for tokens/sec metric
 *   - Timing fields unchanged (CLOCK_MONOTONIC)
 */

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>

typedef enum {
    JOB_STATUS_PENDING = 0,
    JOB_STATUS_RUNNING,
    JOB_STATUS_DONE,
    JOB_STATUS_ERROR
} JobStatus;

typedef struct {
    uint64_t    job_id;

    char       *prompt;
    int         max_tokens;
    float       temperature;

    char       *output;
    size_t      output_len;
    size_t      output_cap;

    int         n_tokens_generated;  /* actual tokens produced */

    JobStatus   status;
    char        error_msg[512];

    struct timespec submit_time;
    struct timespec start_time;
    struct timespec end_time;

    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} InferenceJob;

InferenceJob *job_create(uint64_t job_id, const char *prompt,
                         int max_tokens, float temperature);
void          job_destroy(InferenceJob *job);

int  job_append_output(InferenceJob *job, const char *fragment, size_t len);

void job_mark_running(InferenceJob *job);
void job_mark_done(InferenceJob *job);
void job_mark_error(InferenceJob *job, const char *reason);

void job_wait(InferenceJob *job);

double job_queue_time_ms(const InferenceJob *job);
double job_exec_time_ms(const InferenceJob *job);
double job_total_time_ms(const InferenceJob *job);
double job_tokens_per_sec(const InferenceJob *job);

const char *job_status_str(JobStatus status);

#endif /* JOB_H */
