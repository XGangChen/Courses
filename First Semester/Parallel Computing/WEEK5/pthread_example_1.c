#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

int sum;
void*runner(void*param);

int main(int argc, char*argv[]){
    pthread_attr_t attr;
    pthread_t tid;

    if(argc!=2){
        printf("using command %s <integer>\n", argv[1]);
        exit(1);
    }

    pthread_attr_init(&attr);
    pthread_create(&tid, &attr, runner, argv[1]);
    pthread_join(tid, NULL);

    printf("sum: %d\n", sum);

    return 0;
}

void *runner(void*param){
    int i, upper = atoi((char*)param);
    sum = 0;
    for(i=1; i<=upper; i++){
        sum = sum + i;
    }
    pthread_exit(0);
}

void printMsg(char* msg) {
    printf("%s\n", msg);
 }
 int main(int argc, char** argv) {
    pthread_t thrdID;
    printf("creating a new thread\n");
    pthread_create(&thrdID, NULL, (void*)printMsg, argv[1]);
    printf("created thread %d\n", (int)thrdID);
    pthread_join(thrdID, NULL);
    return 0;
 }