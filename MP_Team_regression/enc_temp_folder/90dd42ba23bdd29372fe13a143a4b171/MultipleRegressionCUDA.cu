#include "kernelCall.h"

#define HEIGHT 40

__global__ void first(double *_x, double *_y, int cols, double** B)
{
	int bCol  = threadIdx.x;
	int bRow = threadIdx.y;

	double bSum = 0;
	double ySum = 0;
	double x1, x2;
	//__shared__ float subX[numStats]{1};

	for (int i = 0; i < numRowsInput; i++) {
		////make B
		//if (bCol == 0) subX[bRow + 1] = _x[i * cols + bRow];
		//__syncthreads();
		//bSum += subX[bRow] * subX[bCol];
		x1 = (bRow == 0) ? 1 : _x[i * cols + bRow-1];
		x2 = (bCol == 0) ? 1 : _x[i * cols + bCol-1];
		bSum += x1 * x2;

		////make Y
		ySum += x1 * _y[i];
	}
	
	printf("[%d, %d][%f, %f]\n", bCol, bRow, bSum, ySum);
	B[bRow][bCol] = bSum;
	printf("asd\n");
	if (bCol == 0)
		B[bRow][cols+1] = ySum;
	printf("zxcv\n");
	
	//Summation done
	//Gaussian elimination start
	if (bRow != 0 || bCol != 0)
		return;
	// Pivotisation of the B matrix.
	double* tmp;
	for (int i = 0; i < cols+1; ++i)
		for (int k = i + 1; k < cols+1; ++k)
			if (B[i][i] < B[k][i]) {
				tmp = B[i];
				B[i] = B[k];
				B[k] = tmp;
			}
}

__global__ void first_1(double *_x, double *_y, int cols, double** B)
{
	int tID = BLOCK_TID_2D;
	int bRow = threadIdx.x;
	int bCol = threadIdx.y;
	
	__shared__ double subX[HEIGHT][16][16];

	for (int bID = 0; bID < ceil((float)numRowsInput / HEIGHT); bID++) {
		int a=0;
		for (int i = 0; i < numRowsInput; i++)
			a += 1;
	}
}

__global__ void second(int cols, double** B, double* coeffs) {
	int cp1 = cols + 1;
	int bCol = threadIdx.x;
	int bRow = threadIdx.y;

	for (int i = 0; i < cols; ++i) {
		if(bRow >= i)
			B[bRow][bCol] -= (B[i][bCol] * B[bRow][i]) / B[i][i];
		__syncthreads();
	}

	if (bRow != 0 || bCol != 0)
		return;

	for (int i = cols; i >= 0; --i) {
		double reduc = B[i][cp1]; 
		for (int j = i; j < cp1; ++j)
			reduc -= B[i][j] * coeffs[j];
		coeffs[i] = reduc / B[i][i];
	}

}


void kernelCall(double* _x, double* _y, double* _coeffs, int cols, double** B) {
	//DS_timer timer(3);
	//timer.setTimerName(0, (char*)"total");
	//timer.setTimerName(1, (char*)"1");
	//timer.setTimerName(2, (char*)"2");

	//cudaStream_t stream[2];

	//LOOP_I(2)
	//	cudaStreamCreate(&stream[i]);
	int n = cols + 1;

	dim3 firstBlock(n, n);
	first << <1, firstBlock>> > (_x, _y, cols, B);
	
	cudaDeviceSynchronize();

	dim3 secondBlock(n, n-1);
	second << <1, secondBlock>> > (cols, B, _coeffs);

	cudaDeviceSynchronize();

	//LOOP_I(2)
	//	cudaStreamDestroy(stream[i]);
}