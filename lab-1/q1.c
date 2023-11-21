#include<stdio.h>
#include <mpi.h>
#include<math.h>
int main(int argc , char ** argv) {

	int rank,size;
	
	MPI_Init(& argc, & argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	
	printf("%f \n",pow(3,rank));
	MPI_Finalize();
	return 0;
	
}