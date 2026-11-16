/* Dedicated worker lifetime guard. No Python API, callbacks, or GIL dependency.
 * Caller owns read_fd through return. Forking after arming is unsupported.
 * The detached thread owns only its CLOEXEC duplicate; the caller retains read_fd.
 */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

static pthread_mutex_t arm_mutex = PTHREAD_MUTEX_INITIALIZER;
static int armed = 0;

static void *watch_lifetime(void *context)
{
    const int descriptor = (int)(intptr_t)context;
    unsigned char value;
    ssize_t result;
    do {
        result = read(descriptor, &value, sizeof(value));
    } while (result < 0 && errno == EINTR);
    /* EOF, unexpected data, and permanent read errors all revoke authority.
     * _exit bypasses language cleanup, handlers, and interpreter locks.
     */
    _exit(125);
}

int kwip_lifetime_arm(int read_fd)
{
    struct stat info;
    int flags;
    int duplicate;
    int error;
    pthread_attr_t attributes;
    pthread_t thread;

    if (read_fd <= 2) return EBADF;
    flags = fcntl(read_fd, F_GETFL);
    if (flags < 0) return errno;
    if (fstat(read_fd, &info) != 0) return errno;
    if ((flags & O_ACCMODE) != O_RDONLY || !S_ISFIFO(info.st_mode)) return EINVAL;

    error = pthread_mutex_lock(&arm_mutex);
    if (error != 0) return error;
    if (armed) {
        (void)pthread_mutex_unlock(&arm_mutex);
        return EALREADY;
    }
    duplicate = fcntl(read_fd, F_DUPFD_CLOEXEC, 3);
    if (duplicate < 0) {
        error = errno;
        (void)pthread_mutex_unlock(&arm_mutex);
        return error;
    }
    error = pthread_attr_init(&attributes);
    if (error != 0) {
        (void)close(duplicate);
        (void)pthread_mutex_unlock(&arm_mutex);
        return error;
    }
    error = pthread_attr_setdetachstate(&attributes, PTHREAD_CREATE_DETACHED);
    if (error == 0) {
        error = pthread_create(&thread, &attributes, watch_lifetime,
                               (void *)(intptr_t)duplicate);
    }
    (void)pthread_attr_destroy(&attributes);
    if (error != 0) {
        (void)close(duplicate);
        (void)pthread_mutex_unlock(&arm_mutex);
        return error;
    }
    armed = 1;
    (void)pthread_mutex_unlock(&arm_mutex);
    return 0;
}
