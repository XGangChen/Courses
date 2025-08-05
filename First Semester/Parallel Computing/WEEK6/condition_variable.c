#include <stdio.h>
#include <pthread.h>

pthread_mutex_t mutex;
pthread_cond_t cond;
int x = 5;

void action(void) {
	pthread_mutex_lock(&mutex);
	if(x!=0){
		pthread_cond_wait(&cond, &mutex);
	}
	pthread_mutex_unlock(&mutex);
	printf("Take action!\n");

}	

void counter(void) {
	pthread_mutex_lock(&mutex);
	while(x!=0){
		x--;
		printf("x:%d\n",x);
	}
	if(x==0){
		pthread_cond_signal(&cond);
	}
	pthread_mutex_unlock(&mutex);
	printf("counter release!\n");
}

int main() {
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	pthread_t p;

	pthread_create(&p, NULL, (void*)counter, NULL);  
	action();
	return 0;
}