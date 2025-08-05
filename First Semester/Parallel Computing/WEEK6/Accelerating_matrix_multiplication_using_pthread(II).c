#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 100

int A [N][N];
int B [N][N];
int C [N][N];
int goldenC [N][N];
struct v {
   int i; /* row */
   //int j; /* column */
};

void *runner(void *param); /* the thread */

int main(int argc, char *argv[]) {
        int i, j, k;
        pthread_t tid[N];       //Thread ID
        pthread_attr_t attr[N]; //Set of thread attributes
        struct timespec t_start, t_end;
        double sequentialelapsedTime;
        double parallelelapsedTime;

        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++) {
                        A[i][j] = rand()%100;
                        B[i][j] = rand()%100;
                }
        }
        // start time
        clock_gettime( CLOCK_REALTIME, &t_start);

        for(i = 0; i < N; i++) {
                 //Assign a row for each thread
                struct v *data = (struct v *) malloc(sizeof(struct v));
                data->i = i;
               
                //Get the default attributes
                pthread_attr_init(&attr[i]);
                //Create the thread
                pthread_create(&tid[i],&attr[i],runner,data);

        }

        for(i = 0; i < N; i++) {
                pthread_join(tid[i], NULL);
        }
        // stop time
        clock_gettime( CLOCK_REALTIME, &t_end);
        // compute and print the elapsed time in millisec
        parallelelapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
        parallelelapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
        printf("Parallel elapsedTime: %lf ms\n", parallelelapsedTime);
        // start time
        clock_gettime( CLOCK_REALTIME, &t_start);
        for(i = 0; i < N; i++) {
                for(j = 0; j < N; j++) {
                        for(k=0; k< N; k++){
                                goldenC[i][j]+=A[i][k] * B[k][j];
                        }
                }
        }
        // stop time
        clock_gettime( CLOCK_REALTIME, &t_end);

        // compute and print the elapsed time in millisec
        sequentialelapsedTime = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
        sequentialelapsedTime += (t_end.tv_nsec - t_start.tv_nsec) / 1000000.0;
        printf("sequential elapsedTime: %lf ms\n", sequentialelapsedTime);

        printf("Speedup %.2lf\n", sequentialelapsedTime/parallelelapsedTime);
        int pass = 1;
        for(i = 0; i < N; i++) {
                for(j = 0; j < N; j++) {
                        if(goldenC[i][j]!=C[i][j]){
                                pass = 0;
                        }
                }
        }

        if(pass==1){
                printf("Test pass!\n");
        }else{
                printf("Test failed!\n");
        }

        return 0;
}

void *runner(void *param) {
        struct v *data = param;
        int row = data->i;
        for (int col = 0; col < N; col++) {
                C[row][col] = 0;
                for (int k = 0; k < N; k++) {
                        C[row][col] += A[row][k] * B[k][col];
                }
        }
        pthread_exit(0);
}