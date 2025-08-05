#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 100000000
#define MAX_ITERS 100
#define ALPHA 0.01f

__device__ void forward_kinematics(const float* q, float* pos_out) {
    pos_out[0] = q[0] + q[1];
    pos_out[1] = q[2] - q[3];
    pos_out[2] = q[4] * q[5];
}

__device__ void batched_matrix_layer(const float* q_in, float* q_out) {
    for (int i = 0; i < 6; i++) {
        q_out[i] = 0;
        for (int j = 0; j < 6; j++) {
            q_out[i] += 0.01f * (i + j) * q_in[j];
        }
    }
}

__device__ void jacobian_transpose_ik(float* target, float* q_init, float* q_out) {
    float q[6];
    for (int i = 0; i < 6; i++) q[i] = q_init[i];

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        float pos[3];
        forward_kinematics(q, pos);

        float err[3] = {
            target[0] - pos[0],
            target[1] - pos[1],
            target[2] - pos[2]
        };

        for (int i = 0; i < 6; i++) {
            float update = 0.1f * (err[0] + err[1] + err[2]);
            q[i] += ALPHA * update;
        }
    }

    batched_matrix_layer(q, q_out);
}

__global__ void ikernel(float* x, float* y, float* z, float* gt_q, float* out_q, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float pos[3] = { x[i], y[i], z[i] };
    float q_init[6] = { 0 };
    jacobian_transpose_ik(pos, q_init, &out_q[i * 6]);
}

float rmse(const float* a, const float* b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sqrtf(sum / n);
}

int main() {
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_z = (float*)malloc(N * sizeof(float));
    float *h_gt_q = (float*)malloc(N * 6 * sizeof(float));
    float *h_out_q = (float*)malloc(N * 6 * sizeof(float));

    FILE* file = fopen("robot_inverse_kinematics_dataset.csv", "r");
    if (!file) {
        perror("File opening failed");
        return EXIT_FAILURE;
    }
    char line[256];
    int idx = 0;
    fgets(line, sizeof(line), file);
    while (fgets(line, sizeof(line), file) && idx < N) {
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &h_gt_q[idx*6+0], &h_gt_q[idx*6+1], &h_gt_q[idx*6+2],
               &h_gt_q[idx*6+3], &h_gt_q[idx*6+4], &h_gt_q[idx*6+5],
               &h_x[idx], &h_y[idx], &h_z[idx]);
        idx++;
    }
    fclose(file);

    float *d_x, *d_y, *d_z, *d_gt_q, *d_out_q;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMalloc(&d_z, N * sizeof(float));
    cudaMalloc(&d_gt_q, N * 6 * sizeof(float));
    cudaMalloc(&d_out_q, N * 6 * sizeof(float));

    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    ikernel<<<blocks, threads>>>(d_x, d_y, d_z, d_gt_q, d_out_q, N);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("CUDA Jacobian IK with FK+batch matmul took %.4f ms.\n", ms);

    cudaMemcpy(h_out_q, d_out_q, N * 6 * sizeof(float), cudaMemcpyDeviceToHost);

    float total_rmse = 0;
    for (int i = 0; i < idx; i++) {
        total_rmse += rmse(&h_out_q[i*6], &h_gt_q[i*6], 6);
    }
    printf("Average RMSE over %d samples: %.6f\n", idx, total_rmse / idx);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_gt_q); cudaFree(d_out_q);
    free(h_x); free(h_y); free(h_z);
    free(h_gt_q); free(h_out_q);
    return 0;
}
