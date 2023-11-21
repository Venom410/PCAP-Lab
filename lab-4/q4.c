#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 4 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int input_matrix[4][4];
    int output_matrix[4][4];
    int local_row[4];

    if (rank == 0) {
        printf("Enter the elements of the 4x4 matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                scanf("%d", &input_matrix[i][j]);
            }
        }
    }

    MPI_Bcast(input_matrix, 16, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scan(input_matrix,output_matrix,4,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    for(int i=0;i<4;i++)
        printf("%d ", output_matrix[i][rank]);
    
    if (rank == 0) {
        printf("Input matrix:\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("%d ", input_matrix[i][j]);
            }
            printf("\n");
        }

       

        
    }

    MPI_Finalize();

    return 0;
}
