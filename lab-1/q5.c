#include<stdio.h>
#include<mpi.h>
int main(int argc , char ** argv) {

	int rank,size;
	MPI_Init(& argc, & argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	if(rank%2==0){
		
		int rankfact=1;
		for(int i=1;i<=rank;i++)
			rankfact=rankfact*i;
		printf("Factorial of rank %d is %d \n",rank,rankfact );
	}
	else{
		int fibbonacci(int n) {
   			if(n == 0){
      			return 0;
   			} else if(n == 1) {
      			return 1;
   			} else {
      	return (fibbonacci(n-1) + fibbonacci(n-2));
   			}
		}

		 		printf("Fibbonacci of rank %d is %d \n",rank,fibbonacci(rank));
	}

	MPI_Finalize();
	return 0;
	
}