#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100000000 // Adjust based on your dataset size

typedef struct {
    float x, y, z;
} Position;

typedef struct {
    float q[6];
} JointAngles;

// Dummy inverse kinematics model (replace with actual model)
void inverse_kinematics(Position pos, JointAngles* joint) {
    joint->q[0] = 0.5f * pos.x + 0.1f * pos.y - 0.2f * pos.z;
    joint->q[1] = -0.3f * pos.x + 0.4f * pos.y + 0.6f * pos.z;
    joint->q[2] = 0.7f * pos.x - 0.5f * pos.y + 0.3f * pos.z;
    joint->q[3] = 0.2f * pos.x + 0.3f * pos.y - 0.1f * pos.z;
    joint->q[4] = -0.4f * pos.x + 0.2f * pos.y + 0.5f * pos.z;
    joint->q[5] = 0.1f * pos.x - 0.3f * pos.y + 0.4f * pos.z;
}

int main() {
    Position* positions = (Position*)malloc(N * sizeof(Position));
    JointAngles* results = (JointAngles*)malloc(N * sizeof(JointAngles));

    // Load data from CSV
    FILE* file = fopen("robot_inverse_kinematics_dataset.csv", "r");
    if (!file) {
        perror("File opening failed");
        return EXIT_FAILURE;
    }
    char line[256];
    int idx = 0;
    fgets(line, sizeof(line), file); // skip header
    while (fgets(line, sizeof(line), file) && idx < N) {
        float q_dummy[6];
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &q_dummy[0], &q_dummy[1], &q_dummy[2], &q_dummy[3], &q_dummy[4], &q_dummy[5],
               &positions[idx].x, &positions[idx].y, &positions[idx].z);
        idx++;
    }
    fclose(file);

    // Time measurement
    clock_t start = clock();
    for (int i = 0; i < N; i++) {
        inverse_kinematics(positions[i], &results[i]);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Serial Inverse Kinematics completed in %.4f seconds.\n", elapsed);

    free(positions);
    free(results);
    return 0;
}
