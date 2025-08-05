#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; 
int count;

void * thread_run(void *arg){    
	int i;    
	pthread_mutex_lock(&mutex);    

	for (i = 0; i < 3; i++) {        
		printf("[%ld]value of count: %d\n", pthread_self(), ++count);    
	}    

	pthread_mutex_unlock(&mutex);    
	return 0;
}
int main(int argc, char *argv[]){    
	pthread_t thread1, thread2, thread3, thread4;    
	pthread_create(&thread1, NULL, thread_run, 0);    	
	pthread_create(&thread2, NULL, thread_run, 0);    	
	pthread_create(&thread3, NULL, thread_run, 0);    	
	pthread_create(&thread4, NULL, thread_run, 0);    	
	printf("thread1 id: %ld\n", thread1);    
	printf("thread2 id: %ld\n", thread2);    	
	printf("thread3 id: %ld\n", thread3);    	
	printf("thread4 id: %ld\n", thread4);    	
	pthread_join(thread1, 0);    
	pthread_join(thread2, 0);    	
	pthread_join(thread3, 0);    	
	pthread_join(thread4, 0);    	
	pthread_mutex_destroy(&mutex);    
	return 0;
}