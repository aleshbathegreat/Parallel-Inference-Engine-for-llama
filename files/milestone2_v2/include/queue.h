#ifndef QUEUE_H
#define QUEUE_H

/*
 * queue.h — Thread-safe FIFO job queue
 *
 * Implemented as a singly-linked list protected by a mutex.
 * Two condition variables allow producers to wait when the queue is full
 * and consumers to wait when the queue is empty.
 *
 * queue_shutdown() is the clean-stop mechanism: after it is called,
 * blocked queue_pop() calls wake up and return NULL, and queue_push()
 * calls return -1.
 */

#include <pthread.h>
#include <stddef.h>
#include "job.h"

/* ── Internal node (opaque to callers) ─────────────────────────────────── */
typedef struct QueueNode {
    InferenceJob    *job;
    struct QueueNode *next;
} QueueNode;

/* ── Queue structure ────────────────────────────────────────────────────── */
typedef struct {
    QueueNode      *head;         /* Oldest element (next to be popped)   */
    QueueNode      *tail;         /* Newest element                        */
    size_t          size;         /* Current number of elements            */
    size_t          max_size;     /* Capacity limit (0 = unlimited)        */
    int             shutdown;     /* Set to 1 by queue_shutdown()          */

    pthread_mutex_t mutex;
    pthread_cond_t  not_empty;    /* Signalled when an item is pushed      */
    pthread_cond_t  not_full;     /* Signalled when an item is popped      */
} JobQueue;

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

/**
 * queue_create() — Allocate and initialise a new JobQueue.
 *
 * @param max_size  Maximum number of items; 0 means no limit.
 * @return          Heap-allocated queue, or NULL on failure.
 */
JobQueue *queue_create(size_t max_size);

/**
 * queue_destroy() — Drain the queue and free all resources.
 *  Jobs still in the queue are NOT freed; the caller is responsible for them.
 *  Call queue_shutdown() before queue_destroy() if workers may be blocking.
 */
void queue_destroy(JobQueue *q);

/* ── Producer API ───────────────────────────────────────────────────────── */

/**
 * queue_push() — Enqueue a job.
 *
 * Blocks if the queue is at capacity until space becomes available,
 * unless the queue has been shut down.
 *
 * @return  0 on success, -1 if the queue is shut down or on allocation error.
 */
int queue_push(JobQueue *q, InferenceJob *job);

/**
 * queue_try_push() — Non-blocking enqueue attempt.
 *
 * @return  0 on success, -1 if full or shut down.
 */
int queue_try_push(JobQueue *q, InferenceJob *job);

/* ── Consumer API ───────────────────────────────────────────────────────── */

/**
 * queue_pop() — Dequeue the oldest job.
 *
 * Blocks until a job is available or the queue is shut down.
 *
 * @return  Pointer to the job, or NULL if the queue is shut down and empty.
 */
InferenceJob *queue_pop(JobQueue *q);

/* ── Control ────────────────────────────────────────────────────────────── */

/**
 * queue_shutdown() — Signal the queue to stop accepting new jobs.
 *  All blocked queue_pop() and queue_push() calls will unblock.
 *  Existing items remain in the queue and can still be popped.
 */
void queue_shutdown(JobQueue *q);

/* ── Introspection (for stats / logging) ───────────────────────────────── */
size_t queue_size(JobQueue *q);
int    queue_is_shutdown(JobQueue *q);

#endif /* QUEUE_H */
