#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#define N 8

/* count tracks the actual number of elements in the buffer */

int main(int argc, char* argv[]) {
    const int SIZE = N * sizeof(int);
    const char *shm_buffer_name = "shm_buffer";
    const char *shm_in_name = "shm_in";
    const char *shm_out_name = "shm_out";
    const char *shm_count_name = "shm_count";       

    if (argc != 2) {
        printf("Usage: %s [integer]\n", argv[0]);
        exit(1);
    }

    int shm_fd, shm_in, shm_out, shm_count;
    int *buffer, *in, *out, *count;

    /* create the shared memory segment */
    shm_fd = shm_open(shm_buffer_name, O_CREAT | O_RDWR, 0666);
    shm_in = shm_open(shm_in_name, O_CREAT | O_RDWR, 0666);
    shm_out = shm_open(shm_out_name, O_CREAT | O_RDWR, 0666);
    shm_count = shm_open(shm_count_name, O_CREAT | O_RDWR, 0666);

    if (shm_fd == -1 || shm_in == -1 || shm_out == -1 || shm_count == -1) {
        printf("Shared memory open failed\n");
        exit(1);
    }

    /* configure the size of the shared memory segment */
    ftruncate(shm_fd, SIZE);
    ftruncate(shm_in, sizeof(int));
    ftruncate(shm_out, sizeof(int));
    ftruncate(shm_count, sizeof(int));

    /* map the shared memory segment in the address space of the process */
    buffer = (int *)mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    in = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_in, 0);
    out = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_out, 0);
    count = (int *)mmap(0, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, shm_count, 0);

    if (buffer == MAP_FAILED || in == MAP_FAILED || out == MAP_FAILED || count == MAP_FAILED) {
        printf("Memory map failed\n");
        exit(1);
    }

    if (*count == N) {
        printf("Buffer is full!\n");
        exit(1);
    }

    buffer[*in] = atoi(argv[1]);
    printf("Produced buffer[%d] = %d\n", *in, buffer[*in]);
    *in = (*in + 1) % N;

    (*count)++;

    printf("Next in: %d, out: %d, count: %d\n", *in, *out, *count);
    return 0;
}
