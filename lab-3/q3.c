#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char input_string[100];  // Adjust the buffer size as needed
    int local_non_vowels = 0;
    int total_non_vowels = 0;
    int local_str_len;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("Enter a string: ");
        fflush(stdout);
        scanf("%s", input_string);
        int str_len = strlen(input_string);
     local_str_len = str_len / size;
    }

    MPI_Bcast(&local_str_len,1,MPI_INT,0,MPI_COMM_WORLD);

    char *local_substring = (char *)malloc((local_str_len + 1) * sizeof(char));
    MPI_Scatter(input_string, local_str_len, MPI_CHAR, local_substring, local_str_len, MPI_CHAR, 0, MPI_COMM_WORLD);

    local_substring[local_str_len] = '\0';

    for (int i = 0; i < local_str_len; i++) {
        if (tolower(local_substring[i]) != 'a' && tolower(local_substring[i]) != 'e' &&
            tolower(local_substring[i]) != 'i' && tolower(local_substring[i]) != 'o' &&
            tolower(local_substring[i]) != 'u') {
            local_non_vowels++;
        }
    }
    printf("Number of non-vowels found by each process:\n");
        
            printf("Process %d: %d non-vowels\n", rank, local_non_vowels);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_non_vowels, &total_non_vowels, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
        
    if (rank == 0) {
        
        printf("Total number of non-vowels: %d\n", total_non_vowels);
    }

    free(local_substring);

    MPI_Finalize();

    return 0;
}
