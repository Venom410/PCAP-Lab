#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires exactly 3 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int matrix[3][3];
    int searchElement;
    int local_count = 0;
    int global_count = 0;

    if (rank == 0) {
        printf("Enter the elements of the 3x3 matrix:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        
        printf("Enter an element to search for: ");
        scanf("%d", &searchElement);
    }

    MPI_Bcast(&searchElement, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_matrix[3][3];
    MPI_Scatter(matrix, 3*3, MPI_INT, local_matrix, 3*3, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (local_matrix[i][j] == searchElement) {
                local_count++;
            }
        }
    }

    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Number of occurrences of %d in the matrix: %d\n", searchElement, global_count);
    }

    MPI_Finalize();

    return 0;
}
