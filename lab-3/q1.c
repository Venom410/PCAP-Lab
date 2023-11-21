#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>

int factorial(int n)
	{
		if (n == 0)
		return 1;
		else
		return(n * factorial(n-1));
	}

void main(int argc,char **argv){
    
    int rank, size,N,c;
    int arr[10];
    int fact_arr[10];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD , &size);
    MPI_Status status;

    if(rank==0){

    	N=size;
    	printf("Enter %d elements :",N);
    	for (int i = 0; i < N; i++)
    	{
    		scanf("%d",&arr[i]);
    	}

		}	

		MPI_Scatter(arr,1,MPI_INT,&c,1,MPI_INT,0,MPI_COMM_WORLD);
			
		fprintf(stdout,"I have received %d in process %d\n",c,rank);
		fflush(stdout);
		
		c=factorial(c);
			
		MPI_Gather(&c,1,MPI_INT,fact_arr,1,MPI_INT,0,MPI_COMM_WORLD);
			
		if(rank==0)
			{
				fprintf(stdout,"The Result gathered in the root \n");
				
				fflush(stdout);
				
				for(int i=0; i<N; i++)
					
				fprintf(stdout,"%d \t",fact_arr[i]);
					
				fflush(stdout);

				printf("\n");
			}

		MPI_Finalize();
	}