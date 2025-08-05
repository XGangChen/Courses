#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>

sem_t semaphore;
int counter = 0;

void* child() {  
	for(int i = 0;i < 5;++i) {    
		sem_wait(&semaphore);    
		printf("Counter = %d\n", ++counter);    
		sleep(1);  
	}  
	pthread_exit(NULL);
}
int main(void) {  
	sem_init(&semaphore, 0, 0);  
	pthread_t t;  
	pthread_create(&t, NULL, child, NULL);  
	printf("Post 2 jobs.\n");  
	sem_post(&semaphore);  
	sem_post(&semaphore);  
	sleep(4);  
	printf("Post 3 jobs.\n");  
	sem_post(&semaphore);  
	sem_post(&semaphore);  
	sem_post(&semaphore);  
	pthread_join(t, NULL);  
	return 0;
}