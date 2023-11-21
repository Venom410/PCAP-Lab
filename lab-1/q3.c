#include<stdio.h>
#include<mpi.h>
int main(int argc , char ** argv) {

	int a=10,b=5;
	int rank,size;
	MPI_Init(& argc, & argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);

	if(rank==0)
		printf("Sum = %d \n",a+b);
	if(rank==1)
		printf("Difference = %d \n",a-b);
	if(rank==2)
		printf("Product = %d \n",a*b);
	if(rank==3)
		printf("Quotient = %d \n",a/b);

		MPI_Finalize();
	return 0;
	
}