/*
 * logger.c — Thread-safe logging implementation
 */

#include "logger.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

/* ── Internal state ─────────────────────────────────────────────────────── */
static struct {
    FILE           *file;       /* Log file (may be NULL) */
    LogLevel        min_level;
    pthread_mutex_t mutex;
    int             initialised;
} g_logger = {
    .file        = NULL,
    .min_level   = LOG_LEVEL_INFO,
    .initialised = 0
};

/* ── Level labels and ANSI colours (stderr only) ────────────────────────── */
static const char *level_label[] = {
    "DEBUG", "INFO ", "WARN ", "ERROR"
};

static const char *level_colour[] = {
    "\033[0;37m",   /* grey   — DEBUG */
    "\033[0;32m",   /* green  — INFO  */
    "\033[0;33m",   /* yellow — WARN  */
    "\033[0;31m"    /* red    — ERROR */
};

#define COLOUR_RESET "\033[0m"

/* ── Public API ─────────────────────────────────────────────────────────── */

void logger_init(const char *log_file, LogLevel min_level)
{
    pthread_mutex_init(&g_logger.mutex, NULL);
    g_logger.min_level   = min_level;
    g_logger.initialised = 1;

    if (log_file) {
        g_logger.file = fopen(log_file, "a");
        if (!g_logger.file) {
            fprintf(stderr, "[LOGGER] Warning: could not open log file '%s'\n",
                    log_file);
        }
    }
}

void logger_close(void)
{
    if (!g_logger.initialised) return;

    pthread_mutex_lock(&g_logger.mutex);
    if (g_logger.file) {
        fclose(g_logger.file);
        g_logger.file = NULL;
    }
    g_logger.initialised = 0;
    pthread_mutex_unlock(&g_logger.mutex);
    pthread_mutex_destroy(&g_logger.mutex);
}

void logger_log(LogLevel level, const char *file, int line,
                const char *fmt, ...)
{
    if (!g_logger.initialised) return;
    if (level < g_logger.min_level) return;

    /* Format timestamp */
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm_info;
    localtime_r(&ts.tv_sec, &tm_info);

    char ts_buf[32];
    strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d %H:%M:%S", &tm_info);

    /* Extract just the filename (not full path) */
    const char *fname = strrchr(file, '/');
    fname = fname ? fname + 1 : file;

    /* Format the caller's message */
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);

    pthread_mutex_lock(&g_logger.mutex);

    /* ── stderr (with ANSI colour) ── */
    fprintf(stderr, "%s%s.%03ld [%s] (%s:%d) %s%s\n",
            level_colour[level],
            ts_buf,
            (long)(ts.tv_nsec / 1000000),
            level_label[level],
            fname, line,
            msg,
            COLOUR_RESET);

    /* ── Log file (plain text) ── */
    if (g_logger.file) {
        fprintf(g_logger.file, "%s.%03ld [%s] (%s:%d) %s\n",
                ts_buf,
                (long)(ts.tv_nsec / 1000000),
                level_label[level],
                fname, line,
                msg);
        fflush(g_logger.file);
    }

    pthread_mutex_unlock(&g_logger.mutex);
}
