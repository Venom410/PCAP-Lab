#include <mpi.h>
#include <stdio.h>

double average(int c[], int n) {
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += c[i];
    return sum / n;
}


int main(int argc, char *argv[]) {
    int rank, size, N, A[100], c[100], i, M;
    double avg,B[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        N = size;
        fprintf(stdout, "Enter the value of M: ");
        fflush(stdout);
        scanf("%d", &M);
        fprintf(stdout, "Enter %d values:\n", N * M);
        fflush(stdout);
        for (i = 0; i < N * M; i++)
            scanf("%d", &A[i]);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Scatter(A, M, MPI_INT, c, M, MPI_INT, 0, MPI_COMM_WORLD);
    for (i = 0; i < M; i++)
        fprintf(stdout, "I have received %d in process %d\n", c[i], rank);
    fflush(stdout);

    avg = average(c, M);
    MPI_Gather(&avg, 1, MPI_DOUBLE, B, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        fprintf(stdout, "The Result gathered in the root:\n");
        for (i = 0; i < N; i++)
            fprintf(stdout, "%lf\t", B[i]);
        fprintf(stdout, "\n");
        double Sum;
        for(i=0;i<N;i++)
            Sum+=B[i];
        fprintf(stdout,"Total average: %lf",Sum/N);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}