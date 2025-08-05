 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <time.h>
 #include <pthread.h>
 
 #define DOF 6
 #define NUM_ATTEMPTS 100000
 #define N_WAYPOINTS 1000
 #define NUM_THREADS 4  // Adjust as needed
 

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

 typedef struct {
     int start_index;
     int end_index;
 } ThreadData;
 

 void* thread_func(void* arg)
 {
     ThreadData* data = (ThreadData*)arg;
     int start = data->start_index;
     int end   = data->end_index;
 
     for(int i = start; i < end; i++) {
         solve_IK(targets[i], solutions[i]);
     }
     return NULL;
 }

 static double get_time_sec()
 {
     struct timespec ts;
     clock_gettime(CLOCK_MONOTONIC, &ts);
     return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
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
 
     // Prepare thread handles and thread data
     pthread_t threads[NUM_THREADS];
     ThreadData thread_data[NUM_THREADS];
 
     // Compute chunk size
     int chunk_size = N_WAYPOINTS / NUM_THREADS;
     int remainder = N_WAYPOINTS % NUM_THREADS;
 
     // Create the threads
     double start_time = get_time_sec();
 
     int current_start = 0;
     for(int t = 0; t < NUM_THREADS; t++) {
         // Distribute leftover remainder among the first threads
         int extra = (t < remainder) ? 1 : 0;
         int size_for_this_thread = chunk_size + extra;
 
         thread_data[t].start_index = current_start;
         thread_data[t].end_index   = current_start + size_for_this_thread;
         current_start += size_for_this_thread;
 
         pthread_create(&threads[t], NULL, thread_func, (void*)&thread_data[t]);
     }
 
     // Join threads
     for(int t = 0; t < NUM_THREADS; t++) {
         pthread_join(threads[t], NULL);
     }
 
     double end_time = get_time_sec();
     double elapsed = end_time - start_time;
 
     // Print a few results
     printf("Pthreads results (first 3 waypoints only):\n");
     for(int i = 0; i < 3; i++) {
         printf("Waypoint %d -> Target(%.2f, %.2f, %.2f) -> Joints: ",
                i, targets[i][0], targets[i][1], targets[i][2]);
         for(int j = 0; j < DOF; j++) {
             printf("%.2f ", solutions[i][j]);
         }
         printf("\n");
     }
 
     printf("\nPthreads total time for %d waypoints with %d threads: %.3f s\n",
            N_WAYPOINTS, NUM_THREADS, elapsed);
 
     return 0;
 }
 