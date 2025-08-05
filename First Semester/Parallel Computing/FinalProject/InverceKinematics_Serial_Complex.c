#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100000000
#define MAX_ITERS 100
#define ALPHA 0.01f

typedef struct {
    float x, y, z;
} Position;

typedef struct {
    float q[6];
} JointAngles;

void forward_kinematics(const float* q, float* pos_out) {
    pos_out[0] = q[0] + q[1];
    pos_out[1] = q[2] - q[3];
    pos_out[2] = q[4] * q[5];
}

void batched_matrix_layer(const float* q_in, float* q_out) {
    float weights[6][6];
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++)
            weights[i][j] = 0.01f * (i + j);

    for (int i = 0; i < 6; i++) {
        q_out[i] = 0;
        for (int j = 0; j < 6; j++) {
            q_out[i] += weights[i][j] * q_in[j];
        }
    }
}

void jacobian_transpose_ik(Position target, float* q_init, float* q_out) {
    float q[6];
    for (int i = 0; i < 6; i++) q[i] = q_init[i];

    for (int iter = 0; iter < MAX_ITERS; iter++) {
        float pos[3];
        forward_kinematics(q, pos);
        float err[3] = { target.x - pos[0], target.y - pos[1], target.z - pos[2] };

        for (int i = 0; i < 6; i++) {
            float update = 0.1f * (err[0] + err[1] + err[2]);
            q[i] += ALPHA * update;
        }
    }

    float q_tmp[6];
    batched_matrix_layer(q, q_tmp);
    for (int i = 0; i < 6; i++) q_out[i] = q_tmp[i];
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
    Position* positions = (Position*)malloc(N * sizeof(Position));
    JointAngles* gt_joints = (JointAngles*)malloc(N * sizeof(JointAngles));
    JointAngles* predicted_joints = (JointAngles*)malloc(N * sizeof(JointAngles));

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
               &gt_joints[idx].q[0], &gt_joints[idx].q[1], &gt_joints[idx].q[2],
               &gt_joints[idx].q[3], &gt_joints[idx].q[4], &gt_joints[idx].q[5],
               &positions[idx].x, &positions[idx].y, &positions[idx].z);
        idx++;
    }
    fclose(file);

    clock_t start = clock();
    for (int i = 0; i < idx; i++) {
        float q_init[6] = {0};
        jacobian_transpose_ik(positions[i], q_init, predicted_joints[i].q);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial Jacobian IK with FK+batch matmul took %.4f seconds.\n", elapsed);

    float total_rmse = 0;
    for (int i = 0; i < idx; i++) {
        total_rmse += rmse(predicted_joints[i].q, gt_joints[i].q, 6);
    }
    printf("Average RMSE over %d samples: %.6f\n", idx, total_rmse / idx);

    free(positions);
    free(gt_joints);
    free(predicted_joints);
    return 0;
}
