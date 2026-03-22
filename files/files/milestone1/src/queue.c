/*
 * queue.c — Thread-safe FIFO job queue (linked-list backed)
 *
 * Concurrency model
 * ─────────────────
 *   • A single mutex serialises all queue state mutations.
 *   • `not_empty` wakes consumers blocked in queue_pop().
 *   • `not_full`  wakes producers blocked in queue_push().
 *   • Setting `shutdown = 1` and broadcasting on both CVs ensures
 *     every blocked thread sees the change on its next iteration.
 */

#include "queue.h"
#include "logger.h"

#include <stdlib.h>
#include <string.h>

/* ── Lifecycle ──────────────────────────────────────────────────────────── */

JobQueue *queue_create(size_t max_size)
{
    JobQueue *q = calloc(1, sizeof(*q));
    if (!q) {
        LOG_ERROR("queue_create: out of memory");
        return NULL;
    }

    q->max_size = max_size;
    q->shutdown = 0;

    pthread_mutex_init(&q->mutex,     NULL);
    pthread_cond_init(&q->not_empty,  NULL);
    pthread_cond_init(&q->not_full,   NULL);

    LOG_INFO("Queue created (max_size=%zu)", max_size ? max_size : (size_t)-1);
    return q;
}

void queue_destroy(JobQueue *q)
{
    if (!q) return;

    pthread_mutex_lock(&q->mutex);

    /* Drain remaining nodes (jobs themselves are NOT freed here) */
    QueueNode *node = q->head;
    size_t count = 0;
    while (node) {
        QueueNode *tmp = node->next;
        free(node);
        node = tmp;
        ++count;
    }
    if (count) {
        LOG_WARN("queue_destroy: %zu jobs were still queued and have been "
                 "discarded (not freed)", count);
    }

    q->head = q->tail = NULL;
    q->size = 0;

    pthread_mutex_unlock(&q->mutex);
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);

    free(q);
}

/* ── Internal helpers ───────────────────────────────────────────────────── */

/* Allocate a new node.  Returns NULL on allocation failure. */
static QueueNode *node_alloc(InferenceJob *job)
{
    QueueNode *n = malloc(sizeof(*n));
    if (!n) return NULL;
    n->job  = job;
    n->next = NULL;
    return n;
}

/* Append a node to the tail of the queue.  Caller must hold the mutex. */
static void enqueue_locked(JobQueue *q, QueueNode *node)
{
    if (q->tail) {
        q->tail->next = node;
    } else {
        q->head = node;
    }
    q->tail = node;
    q->size++;
}

/* Remove and return the job at the head of the queue.
   Caller must hold the mutex and must have verified q->size > 0. */
static InferenceJob *dequeue_locked(JobQueue *q)
{
    QueueNode *node = q->head;
    InferenceJob *job = node->job;

    q->head = node->next;
    if (!q->head) q->tail = NULL;
    q->size--;

    free(node);
    return job;
}

/* ── Producer API ───────────────────────────────────────────────────────── */

int queue_push(JobQueue *q, InferenceJob *job)
{
    if (!q || !job) return -1;

    QueueNode *node = node_alloc(job);
    if (!node) {
        LOG_ERROR("queue_push: node allocation failed");
        return -1;
    }

    pthread_mutex_lock(&q->mutex);

    /* Wait until there is space (or shutdown) */
    while (q->max_size > 0 && q->size >= q->max_size && !q->shutdown) {
        LOG_DEBUG("queue_push: queue full (%zu/%zu) — blocking",
                  q->size, q->max_size);
        pthread_cond_wait(&q->not_full, &q->mutex);
    }

    if (q->shutdown) {
        pthread_mutex_unlock(&q->mutex);
        free(node);
        LOG_WARN("queue_push: rejected — queue is shut down");
        return -1;
    }

    enqueue_locked(q, node);
    LOG_DEBUG("queue_push: job #%llu enqueued (queue size=%zu)",
              (unsigned long long)job->job_id, q->size);

    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

int queue_try_push(JobQueue *q, InferenceJob *job)
{
    if (!q || !job) return -1;

    pthread_mutex_lock(&q->mutex);

    if (q->shutdown ||
        (q->max_size > 0 && q->size >= q->max_size)) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }

    QueueNode *node = node_alloc(job);
    if (!node) {
        pthread_mutex_unlock(&q->mutex);
        return -1;
    }

    enqueue_locked(q, node);
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
    return 0;
}

/* ── Consumer API ───────────────────────────────────────────────────────── */

InferenceJob *queue_pop(JobQueue *q)
{
    if (!q) return NULL;

    pthread_mutex_lock(&q->mutex);

    /* Wait until there is something to pop or the queue is shut down */
    while (q->size == 0 && !q->shutdown) {
        pthread_cond_wait(&q->not_empty, &q->mutex);
    }

    if (q->size == 0) {
        /* Shut down and empty */
        pthread_mutex_unlock(&q->mutex);
        return NULL;
    }

    InferenceJob *job = dequeue_locked(q);
    LOG_DEBUG("queue_pop: job #%llu dequeued (queue size=%zu)",
              (unsigned long long)job->job_id, q->size);

    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return job;
}

/* ── Control ────────────────────────────────────────────────────────────── */

void queue_shutdown(JobQueue *q)
{
    if (!q) return;

    pthread_mutex_lock(&q->mutex);
    q->shutdown = 1;
    /* Wake all blocked producers and consumers */
    pthread_cond_broadcast(&q->not_empty);
    pthread_cond_broadcast(&q->not_full);
    pthread_mutex_unlock(&q->mutex);

    LOG_INFO("Queue shut down (%zu jobs remaining)", q->size);
}

/* ── Introspection ──────────────────────────────────────────────────────── */

size_t queue_size(JobQueue *q)
{
    if (!q) return 0;
    pthread_mutex_lock(&q->mutex);
    size_t s = q->size;
    pthread_mutex_unlock(&q->mutex);
    return s;
}

int queue_is_shutdown(JobQueue *q)
{
    if (!q) return 1;
    pthread_mutex_lock(&q->mutex);
    int s = q->shutdown;
    pthread_mutex_unlock(&q->mutex);
    return s;
}
