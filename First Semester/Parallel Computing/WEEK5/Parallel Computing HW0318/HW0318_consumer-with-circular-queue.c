#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define N 8

int main() {
    const int SIZE = N * sizeof(int);
    const char *shm_buffer_name = "shm_buffer";
    const char *shm_in_name = "shm_in";
    const char *shm_out_name = "shm_out";
    const char *shm_count_name = "shm_count";

    int shm_fd, shm_in, shm_out, shm_count;
    int *buffer, *in, *out, *count;

    /* open the shared memory segment */
    shm_fd = shm_open(shm_buffer_name, O_RDWR, 0666);
    shm_in = shm_open(shm_in_name, O_RDWR, 0666);
    shm_out = shm_open(shm_out_name, O_RDWR, 0666);
    shm_count = shm_open(shm_count_name, O_RDWR, 0666);

    if (shm_fd == -1 || shm_in == -1 || shm_out == -1 || shm_count == -1) {
        printf("Shared memory open failed\n");
        exit(1);
    }

    /* map the shared memory segment in the address space of the process */
    buffer = (int *)mmap(0, SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);
    in = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_in, 0);
    out = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_out, 0);
    count = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_count, 0);

    if (buffer == MAP_FAILED || in == MAP_FAILED || out == MAP_FAILED || count == MAP_FAILED) {
        printf("Memory map failed\n");
        exit(1);
    }

    if (*count == 0) {
        printf("Buffer is empty!\n");
        exit(1);
    }

    printf("Consumed buffer[%d] = %d\n", *out, buffer[*out]);
    *out = (*out + 1) % N;
    (*count)--;

    printf("in: %d, next out: %d, count: %d\n", *in, *out, *count);

    return 0;
}
