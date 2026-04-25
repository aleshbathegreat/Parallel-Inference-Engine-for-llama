#ifndef CORRECTNESS_H
#define CORRECTNESS_H

/*
 * correctness.h — Milestone 3 correctness test suite
 *
 * Tests that the engine produces correct, non-corrupted results
 * under concurrent load at various concurrency levels.
 *
 * Test cases:
 *   1. Determinism    — same prompt + greedy sampling = same output every time
 *   2. No corruption  — each job's output matches its prompt (not mixed up)
 *   3. Concurrent load — N threads submit simultaneously, all succeed
 *   4. Queue stress   — submit more jobs than workers, all complete
 *   5. Output isolation — different prompts produce different outputs
 */

#include "engine.h"

typedef struct {
    const char *name;
    int         passed;
    int         failed;
    char        details[512];
} TestResult;

/*
 * run_correctness_suite()
 *
 * Runs all correctness tests against the given engine config template.
 * @param model      Pre-loaded model
 * @param n_workers  Number of workers to use for concurrent tests
 * @return           Number of tests that passed
 */
int run_correctness_suite(struct llama_model *model, int n_workers);

#endif /* CORRECTNESS_H */
