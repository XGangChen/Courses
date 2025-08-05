#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void printMsg(char* msg) {
	static int status = 1234;
	printf("%s\n", msg);
	pthread_exit(&status);//phread_exit is used to return data via pointer
}

int main(int argc, char** argv) {
	pthread_t thrdID;
	int* status = (int*)malloc(sizeof(int));

	printf("creating a new thread\n");
	pthread_create(&thrdID, NULL, (void*)printMsg, argv[1]);
	printf("created thread %d\n", (int)thrdID);
	pthread_join(thrdID, (void**)&status);//pthread_join is used to get return value
	printf("Thread %d exited with status %d\n", (int)thrdID, *status);

	return 0;
}
