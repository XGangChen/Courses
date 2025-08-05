#include <stdio.h>
#include <pthread.h>

#define MAX_SIZE  5
pthread_mutex_t bufLock;
int count;

void producer(char* buf) {
	for(;;) {
		while(count == MAX_SIZE);
		pthread_mutex_lock(&bufLock);
		printf("enter a char:");
		buf[count] = getchar();
		getchar();
		count++;
		pthread_mutex_unlock(&bufLock);
	}
}	

void consumer(char* buf) {
	for(;;) {
		while(count == 0);
		pthread_mutex_lock(&bufLock);
		printf("output buffer:");
		printf("%c\n", buf[count-1]);
		count--;
		pthread_mutex_unlock(&bufLock);
	}
}

int main() {
	char buffer[MAX_SIZE];
	pthread_t p;
	count = 0;
	pthread_mutex_init(&bufLock, NULL);
	pthread_create(&p, NULL, (void*)producer, buffer);  
	consumer(buffer);
	return 0;
}
