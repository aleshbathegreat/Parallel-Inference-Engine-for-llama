#ifndef LOGGER_H
#define LOGGER_H

/*
 * logger.h — Simple thread-safe logging subsystem
 *
 * Supports four severity levels (DEBUG, INFO, WARN, ERROR).
 * Output goes to stderr and, optionally, a log file.
 * All calls are safe to make from multiple threads simultaneously.
 */

#include <stdio.h>

/* ── Log levels ─────────────────────────────────────────────────────────── */
typedef enum {
    LOG_LEVEL_DEBUG = 0,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR
} LogLevel;

/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * logger_init() — Initialise the logger.
 *
 * @param log_file  Path to log file; NULL disables file output.
 * @param min_level Minimum severity that will be emitted.
 */
void logger_init(const char *log_file, LogLevel min_level);

/**
 * logger_close() — Flush and close the logger.
 *  Must be called before the process exits.
 */
void logger_close(void);

/**
 * logger_log() — Emit a formatted log message.
 *  Prefer the macros below rather than calling this directly.
 */
void logger_log(LogLevel level, const char *file, int line,
                const char *fmt, ...)
    __attribute__((format(printf, 4, 5)));

/* ── Convenience macros ─────────────────────────────────────────────────── */
#define LOG_DEBUG(fmt, ...) \
    logger_log(LOG_LEVEL_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    logger_log(LOG_LEVEL_INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) \
    logger_log(LOG_LEVEL_WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) \
    logger_log(LOG_LEVEL_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#endif /* LOGGER_H */
