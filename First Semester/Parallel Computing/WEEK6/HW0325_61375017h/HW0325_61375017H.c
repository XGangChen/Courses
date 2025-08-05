#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>

#define N 100

int a[N][N], b[N][N], bt[N][N], c[N][N], ct[N][N], seq_c[N][N], seq_ct[N][N];
int num_threads;

void* multiply_no_transpose(void* arg) {
    int id = *(int*)arg;
    int start = id * N / num_threads;
    int end = (id + 1) * N / num_threads;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    pthread_exit(0);
}

void* multiply_with_transpose(void* arg) {
    int id = *(int*)arg;
    int start = id * N / num_threads;
    int end = (id + 1) * N / num_threads;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            ct[i][j] = 0;
            for (int k = 0; k < N; k++) {
                ct[i][j] += a[i][k] * bt[j][k];
            }
        }
    }
    pthread_exit(0);
}

int main() {
    struct timespec t_start, t_end;
    double elapsedTime;
    num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            a[i][j] = rand() % 100;
            b[i][j] = rand() % 100;
        }
    }

    // Transpose matrix b -> bt
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            bt[j][i] = b[i][j];

    // Sequential without transpose
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            seq_c[i][j] = 0;
            for (int k = 0; k < N; k++) {
                seq_c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Sequential (no transpose) elapsed time: %lf ms\n", elapsedTime);

    // Sequential with transpose
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            seq_ct[i][j] = 0;
            for (int k = 0; k < N; k++) {
                seq_ct[i][j] += a[i][k] * bt[j][k];
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Sequential (with transpose) elapsed time: %lf ms\n", elapsedTime);

    // Pthread without transpose
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, multiply_no_transpose, &thread_ids[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Pthread (no transpose) elapsed time: %lf ms\n", elapsedTime);

    // Pthread with transpose
    clock_gettime(CLOCK_REALTIME, &t_start);
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, multiply_with_transpose, &thread_ids[i]);
    }
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_REALTIME, &t_end);
    elapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
    elapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
    printf("Pthread (with transpose) elapsed time: %lf ms\n", elapsedTime);

    return 0;
}
