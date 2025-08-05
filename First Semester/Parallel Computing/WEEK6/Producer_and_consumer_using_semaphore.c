#include <stdio.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#define MAX_SIZE  5
sem_t empty, full;

void producer(char* buf) {
   int in = 0;
   for(;;) {
	sem_wait(&empty);
        printf("enter a char:");	
	buf[in] = getchar();
	getchar();
	in = (in + 1) % MAX_SIZE;
	sem_post(&full);
   }
}	
void consumer(char* buf) {
   int out = 0;
   for(;;) {
	sem_wait(&full);
	printf("Output buffer:");
	printf("%c\n", buf[out]);
	out = (out + 1) % MAX_SIZE;
	sem_post(&empty);
   }
}

int main() {
   char buffer[MAX_SIZE];
   pthread_t p;
   sem_init(&empty, 0, MAX_SIZE);
   sem_init(&full, 0, 0);
   pthread_create(&p, NULL, (void*)producer, buffer);  
   consumer(buffer);
   return 0;
}