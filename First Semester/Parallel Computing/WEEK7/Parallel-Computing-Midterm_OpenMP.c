 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 #include <omp.h>
 
 #define DOF 6
 #define NUM_ATTEMPTS 100000
 #define N_WAYPOINTS 1000
 
 static double targets[N_WAYPOINTS][3];
 static double solutions[N_WAYPOINTS][DOF];
 

 void compute_FK(const double joints[DOF], double out_pos[3])
 {
     double link_length = 0.1;
     double x = 0.0, y = 0.0, z = 0.0;
 
     for(int i = 0; i < DOF; i++) {
         x += link_length * cos(joints[i]);
         y += link_length * sin(joints[i]);
         if(i % 2 == 0) {
             z += link_length;
         }
     }
 
     out_pos[0] = x;
     out_pos[1] = y;
     out_pos[2] = z;
 }
 

 void solve_IK(const double target[3], double solution[DOF])
 {
     double best_joints[DOF];
     double best_error = 1e9;
 
     for(int attempt = 0; attempt < NUM_ATTEMPTS; attempt++) {
         double test_joints[DOF];
         for(int j = 0; j < DOF; j++) {
             test_joints[j] = ((double)rand() / RAND_MAX) * 2.0 * M_PI - M_PI;
         }
 
         double fk_pos[3];
         compute_FK(test_joints, fk_pos);
 
         double dx = fk_pos[0] - target[0];
         double dy = fk_pos[1] - target[1];
         double dz = fk_pos[2] - target[2];
         double error = sqrt(dx*dx + dy*dy + dz*dz);
 
         if(error < best_error) {
             best_error = error;
             for(int k = 0; k < DOF; k++) {
                 best_joints[k] = test_joints[k];
             }
         }
     }
 
     for(int i = 0; i < DOF; i++) {
         solution[i] = best_joints[i];
     }
 }
 
 int main(void)
 {
     srand(42);
     // Generate random targets
     for(int i = 0; i < N_WAYPOINTS; i++) {
         targets[i][0] = (double)rand() / RAND_MAX; 
         targets[i][1] = (double)rand() / RAND_MAX; 
         targets[i][2] = (double)rand() / RAND_MAX; 
     }
 
     double start_time = omp_get_wtime();
 
     // Parallelize the loop over waypoints.
     // By default, each thread calls solve_IK(...) on its subset.
     #pragma omp parallel for
     for(int i = 0; i < N_WAYPOINTS; i++) {
         solve_IK(targets[i], solutions[i]);
     }
 
     double end_time = omp_get_wtime();
     double elapsed = end_time - start_time;
 
     // Print a few results
     printf("OpenMP results (first 3 waypoints only):\n");
     for(int i = 0; i < 3; i++) {
         printf("Waypoint %d -> Target(%.2f, %.2f, %.2f) -> Joints: ",
                i, targets[i][0], targets[i][1], targets[i][2]);
         for(int j = 0; j < DOF; j++) {
             printf("%.2f ", solutions[i][j]);
         }
         printf("\n");
     }
 
     // Number of threads used:
     int used_threads = 1;
     #pragma omp parallel
     {
         // Count max number of threads in parallel region
         used_threads = omp_get_num_threads();
     }
 
     printf("\nOpenMP total time for %d waypoints with %d threads: %.3f s\n",
            N_WAYPOINTS, used_threads, elapsed);
 
     return 0;
 }
 