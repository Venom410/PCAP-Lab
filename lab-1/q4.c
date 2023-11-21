#include <stdio.h>
#include <mpi.h>
	int main(int argc, char **argv){
		int rank, size;
		char str[6] = "HELLO";

	MPI_Init(&argc, &argv);	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	str[rank] = str[rank]+32;

	printf("%s\n",str);

	MPI_Finalize();

	return 0;

}