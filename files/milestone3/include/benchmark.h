#ifndef BENCHMARK_H
#define BENCHMARK_H

/*
 * benchmark.h — Milestone 3 benchmarking and analysis module
 *
 * Provides:
 *   - ScalingResult: metrics for one concurrency level
 *   - LatencyDist:   p50/p95/p99 latency distribution
 *   - run_scaling_sweep(): runs 1..MAX_WORKERS and collects all metrics
 *   - print_full_report(): prints the complete analysis table
 */

#include <stdint.h>
#include "engine.h"

/* Maximum workers tested in the sweep */
#define MAX_SWEEP_WORKERS 4

/* ── Latency distribution (per concurrency level) ───────────────────────── */
typedef struct {
    double p50_ms;    /* median latency                   */
    double p95_ms;    /* 95th percentile                  */
    double p99_ms;    /* 99th percentile                  */
    double min_ms;
    double max_ms;
    double mean_ms;
} LatencyDist;

/* ── Result for one concurrency level ───────────────────────────────────── */
typedef struct {
    int      n_workers;
    int      jobs_done;
    int      jobs_failed;
    double   wall_ms;          /* total elapsed time               */
    double   jobs_per_sec;     /* throughput                       */
    double   tokens_per_sec;   /* token throughput                 */
    uint64_t total_tokens;
    LatencyDist latency;       /* exec time distribution           */
    double   avg_queue_ms;     /* mean time spent waiting in queue */
    double   speedup;          /* vs n_workers=1 baseline          */
    double   efficiency;       /* speedup / n_workers * 100        */
} ScalingResult;

/* ── Run the full scaling sweep ─────────────────────────────────────────── */
/*
 * run_scaling_sweep()
 *
 * For each worker count in [1, 2, 4] (up to MAX_SWEEP_WORKERS):
 *   1. Creates an engine with that many workers
 *   2. Submits all prompts asynchronously (queue fills instantly)
 *   3. Collects per-job latencies
 *   4. Computes speedup and efficiency vs the 1-worker baseline
 *
 * @param model       Pre-loaded llama_model (shared, not reloaded)
 * @param prompts     Array of prompt strings
 * @param n_prompts   Number of prompts
 * @param max_tokens  Token limit per job
 * @param results     Output array — must have MAX_SWEEP_WORKERS entries
 * @return            Number of levels actually run
 */
int run_scaling_sweep(struct llama_model *model,
                      const char **prompts, int n_prompts,
                      int max_tokens,
                      ScalingResult results[MAX_SWEEP_WORKERS]);

/* ── Print the full benchmark report ────────────────────────────────────── */
void print_full_report(const ScalingResult *results, int n_results,
                       int n_cpus);

/* ── Compute latency distribution from an array of ms values ───────────── */
LatencyDist compute_latency_dist(double *ms_values, int n);

#endif /* BENCHMARK_H */
