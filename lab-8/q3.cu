#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
__global__ void compute(int *a, int *res,int m, int n){
	int row=threadIdx.x+blockIdx.x*blockDim.x;
	
	for(int i=0;i<n;i++){
		if(i==0||i==n-1||row==0||row==m-1)
			res[n*row+i]=a[n*row+i];
		else{
			int bin=0;
			int copy=a[n*row+i];
			int k=1;
			int comp=0;
			while(copy!=0){
				bin+=k*(copy%2);
				copy/=2;
				k*=10;
			}
			k=1;
			while(bin!=0){
				int rem=bin%10;
				comp+=rem==1?0:k;
				bin/=10;
				k*=10;
			}
			res[n*row+i]=comp;
		}
	
	}
}
int main(){
	int m,n;
	printf("enter dimensions of matrix\n");
	scanf("%d%d",&m,&n);
	int a[m*n];
	int *d_a,*d_res;
	printf("enter the matrix\n");
	for(int i=0;i<m*n;i++)
		scanf("%d",&a[i]);
	cudaMalloc((void**)&d_a,sizeof(int)*m*n);
	cudaMalloc((void**)&d_res,sizeof(int)*m*n);
	int *res=(int*)malloc(sizeof(int)*m*n);
	cudaMemcpy(d_a,a,sizeof(int)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_res,res,sizeof(int)*m*n,cudaMemcpyHostToDevice);
	compute<<<1,m>>>(d_a,d_res,m,n);
	cudaMemcpy(res,d_res,sizeof(int)*m*n,cudaMemcpyDeviceToHost);
	printf("resultant matrix: \n");
	for(int i=0;i<m*n;i++){
		printf("%d ",res[i]);
		if((i+1)%n==0)
			printf("\n");
	}
}