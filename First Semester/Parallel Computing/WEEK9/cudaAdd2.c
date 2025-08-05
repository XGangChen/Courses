// nvcc cudaAdd2.cu -o cudaAdd2 -Xcompiler -fopenmp

#include <stdio.h>

#include <stdlib.h>

#include <time.h>

#include <cuda_runtime.h>

#include <omp.h>

#define N 1024*1024



__global__ void add( int *a, int *b, int *c ){

	int tid = blockIdx.x;

	if (tid < N) {

        c[tid] = a[tid] + b[tid];

    }

}



int main( void ) {

	int *a, *b, *c, *goldenC;

	int *dev_a, *dev_b, *dev_c;

	cudaEvent_t start, stop;

    float elapsedTime;


	// allocate the memory on the CPU

	a = (int*)malloc( N * sizeof(int) );

	b = (int*)malloc( N * sizeof(int) );

	c = (int*)malloc( N * sizeof(int) );

	goldenC = (int*)malloc( N * sizeof(int) );


	// allocate the memory on the GPU

	cudaMalloc( (void**)&dev_a, N * sizeof(int) );

	cudaMalloc( (void**)&dev_b, N * sizeof(int) );

	cudaMalloc( (void**)&dev_c, N * sizeof(int) );

	// fill the arrays 'a' and 'b' on the CPU

	srand ( time(NULL) );

	for (int i=0; i<N; i++) {

		a[i] = rand()%256;

		b[i] = rand()%256;

	}

	// copy the arrays 'a' and 'b' to the GPU

	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_b, b, N * sizeof(int),cudaMemcpyHostToDevice);



    // Create events

    cudaEventCreate(&start);

    cudaEventCreate(&stop);

	// Record the start event

    cudaEventRecord(start, 0);


	add<<<N,1>>>( dev_a, dev_b, dev_c );


	// Record the stop event

    cudaEventRecord(stop, 0);


	// Wait for the stop event to complete

    cudaEventSynchronize(stop);



    // Calculate elapsed time

    cudaEventElapsedTime(&elapsedTime, start, stop);



    printf("Kernel execution time: %.4f ms\n", elapsedTime);



    // Clean up

    cudaEventDestroy(start);

    cudaEventDestroy(stop);


	// copy the array 'c' back from the GPU to the CPU

	cudaMemcpy( c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	// verify that the GPU did the work we requested


	double omp_start = omp_get_wtime();


	#pragma omp parallel for

	for (int i=0; i<N; i++) {

		goldenC[i] = a[i] + b[i];

	}	

    

	double omp_end = omp_get_wtime();



    printf("OpenMP elapsed time: %.6f ms\n", (omp_end - omp_start)*1000);	





	bool success = true;

	for (int i=0; i<N; i++) {

		if ( goldenC[i] != c[i]) {

			printf( "Error:  %d + %d != %d\n", a[i], b[i], c[i] );

			success = false;

		}

	}

	if (success)

		printf("We did it!\n");

	// free the memory allocated on the GPU

	cudaFree( dev_a );

	cudaFree( dev_b );

	cudaFree( dev_c );



	return 0;

}

