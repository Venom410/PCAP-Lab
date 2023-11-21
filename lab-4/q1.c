#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int factorial(int n) {
    if (n <= 1)
        return 1;
    else
        return n * factorial(n - 1);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 5;  // Number of terms
    int local_sum = 0;

    if (rank == 0) {
        if (argc > 1) {
            n = atoi(argv[1]);
        } else {
            printf("Enter the value of n: ");
            scanf("%d", &n);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n < 0) {
        if (rank == 0) {
            fprintf(stderr, "Number of terms cannot be negative.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = rank + 1; i <= n; i += size) {
        local_sum += factorial(i);
    }

    int global_sum;
    MPI_Scan(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == size - 1) {
        printf("Rank %d: The sum of factorials up to %d is %d\n", rank, n, global_sum);
    }

    MPI_Finalize();

    return 0;
}
