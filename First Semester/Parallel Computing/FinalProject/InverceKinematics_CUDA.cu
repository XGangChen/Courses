#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 100000000

typedef struct {
    float x, y, z;
} Position;

typedef struct {
    float q[6];
} JointAngles;

__device__ void inverse_kinematics_device(Position pos, JointAngles* joint) {
    joint->q[0] = 0.5f * pos.x + 0.1f * pos.y - 0.2f * pos.z;
    joint->q[1] = -0.3f * pos.x + 0.4f * pos.y + 0.6f * pos.z;
    joint->q[2] = 0.7f * pos.x - 0.5f * pos.y + 0.3f * pos.z;
    joint->q[3] = 0.2f * pos.x + 0.3f * pos.y - 0.1f * pos.z;
    joint->q[4] = -0.4f * pos.x + 0.2f * pos.y + 0.5f * pos.z;
    joint->q[5] = 0.1f * pos.x - 0.3f * pos.y + 0.4f * pos.z;
}

__global__ void ikernel(Position* pos, JointAngles* result, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        inverse_kinematics_device(pos[i], &result[i]);
    }
}

int main() {
    Position* h_pos = (Position*)malloc(N * sizeof(Position));
    JointAngles* h_result = (JointAngles*)malloc(N * sizeof(JointAngles));

    FILE* file = fopen("robot_inverse_kinematics_dataset.csv", "r");
    if (!file) {
        perror("File opening failed");
        return EXIT_FAILURE;
    }
    char line[256];
    int idx = 0;
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file) && idx < N) {
        float q_dummy[6];
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &q_dummy[0], &q_dummy[1], &q_dummy[2], &q_dummy[3], &q_dummy[4], &q_dummy[5],
               &h_pos[idx].x, &h_pos[idx].y, &h_pos[idx].z);
        idx++;
    }
    fclose(file);

    Position* d_pos;
    JointAngles* d_result;
    cudaMalloc((void**)&d_pos, N * sizeof(Position));
    cudaMalloc((void**)&d_result, N * sizeof(JointAngles));

    cudaMemcpy(d_pos, h_pos, N * sizeof(Position), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    ikernel<<<blocksPerGrid, threadsPerBlock>>>(d_pos, d_result, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Inverse Kinematics completed in %.4f ms.\n", milliseconds);

    cudaMemcpy(h_result, d_result, N * sizeof(JointAngles), cudaMemcpyDeviceToHost);

    cudaFree(d_pos);
    cudaFree(d_result);
    free(h_pos);
    free(h_result);
    return 0;
}
