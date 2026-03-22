#ifndef JOB_H
#define JOB_H

/*
 * job.h — Inference job representation
 *
 * An InferenceJob encapsulates everything the engine needs to execute a single
 * inference request: the prompt, generation parameters, timing metadata, and
 * the output buffer.  A pair of mutex + condition variable lets callers block
 * until the job finishes (synchronous mode) while still allowing the job to be
 * placed on a shared queue (asynchronous mode).
 */

#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>

/* ── Job status ─────────────────────────────────────────────────────────── */
typedef enum {
    JOB_STATUS_PENDING = 0,   /* Waiting in the queue                */
    JOB_STATUS_RUNNING,       /* Worker thread is executing this job  */
    JOB_STATUS_DONE,          /* Inference completed successfully      */
    JOB_STATUS_ERROR          /* An error occurred during inference    */
} JobStatus;

/* ── Job structure ──────────────────────────────────────────────────────── */
typedef struct {
    /* Identity */
    uint64_t    job_id;

    /* Input */
    char       *prompt;           /* Null-terminated UTF-8 prompt string   */
    int         max_tokens;       /* Maximum number of tokens to generate  */
    float       temperature;      /* Sampling temperature (0 = greedy)     */

    /* Output — heap-allocated, grown as needed */
    char       *output;
    size_t      output_len;       /* Current string length (excl. NUL)     */
    size_t      output_cap;       /* Allocated capacity in bytes           */

    /* Status & error */
    JobStatus   status;
    char        error_msg[512];

    /* Timing (CLOCK_MONOTONIC) */
    struct timespec submit_time;  /* Set when job is enqueued              */
    struct timespec start_time;   /* Set when worker begins execution      */
    struct timespec end_time;     /* Set when job transitions to DONE/ERR  */

    /* Synchronisation — allows callers to block until completion */
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
} InferenceJob;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

/**
 * job_create() — Allocate and initialise a new InferenceJob.
 *
 * @param job_id       Unique identifier (managed by the engine).
 * @param prompt       Input prompt; copied internally.
 * @param max_tokens   Maximum generation tokens (clamped to a reasonable max).
 * @param temperature  Sampling temperature.
 * @return             Heap-allocated job, or NULL on allocation failure.
 */
InferenceJob *job_create(uint64_t job_id, const char *prompt,
                         int max_tokens, float temperature);

/**
 * job_destroy() — Free all resources owned by a job.
 *  The caller must ensure no other thread is waiting on this job.
 */
void job_destroy(InferenceJob *job);

/* ── Output helpers ─────────────────────────────────────────────────────── */

/**
 * job_append_output() — Append a UTF-8 string fragment to the output buffer.
 *  The buffer is grown with realloc() as needed.
 * @return 0 on success, -1 on allocation failure.
 */
int job_append_output(InferenceJob *job, const char *fragment, size_t len);

/* ── Status transitions (called by the engine worker) ──────────────────── */
void job_mark_running(InferenceJob *job);
void job_mark_done(InferenceJob *job);
void job_mark_error(InferenceJob *job, const char *reason);

/* ── Waiting (called by submitters) ─────────────────────────────────────── */

/**
 * job_wait() — Block the calling thread until the job reaches DONE or ERROR.
 */
void job_wait(InferenceJob *job);

/* ── Timing helpers ─────────────────────────────────────────────────────── */

/** Time spent waiting in the queue (submit → start), in milliseconds. */
double job_queue_time_ms(const InferenceJob *job);

/** Time spent executing inference (start → end), in milliseconds. */
double job_exec_time_ms(const InferenceJob *job);

/** Total turnaround time (submit → end), in milliseconds. */
double job_total_time_ms(const InferenceJob *job);

/* ── Utility ────────────────────────────────────────────────────────────── */
const char *job_status_str(JobStatus status);

#endif /* JOB_H */
