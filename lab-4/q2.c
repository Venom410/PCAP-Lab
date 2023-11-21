#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>

int main(int argc, char **argv){
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD , &rank);
    MPI_Comm_size( MPI_COMM_WORLD , &size);

    double dx = 1/((double)size);
    double fx = (4/(1 + (dx * rank)*(dx * rank)));
    double area = fx * dx;
    double integral;
    MPI_Reduce(&area, &integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        printf("pi = %lf\n", integral);
    }
    MPI_Finalize();
}