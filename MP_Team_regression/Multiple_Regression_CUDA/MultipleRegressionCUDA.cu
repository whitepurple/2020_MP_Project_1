#include "kernelCall.h"

#define HEIGHT 40
#define _id(y, x, cols) ((y)*(cols)+(x))

__global__ void first(double *_x, double *_y, int cols, double* B)
{
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol  = threadIdx.x;
	int bRow = threadIdx.y;

	double bSum = 0;
	double ySum = 0;
	double x1, x2;
	//__shared__ float subX[numStats]{1};

	int bRowm1 = bRow - 1;
	int bColm1 = bCol - 1;

	for (int i = 0; i < numRowsInput; i++) {
		////make B
		//if (bCol == 0) subX[bRow + 1] = _x[i * cols + bRow];
		//__syncthreads();
		//bSum += subX[bRow] * subX[bCol];
		x1 = (bRow == 0) ? 1 : _x[i * cols + bRowm1];
		x2 = (bCol == 0) ? 1 : _x[i * cols + bColm1];
		bSum += x1 * x2;

		////make Y
		ySum += x1 * _y[i];
	}

	B[_id(bRow, bCol, cp2)] = bSum;
	printf("[[%d,%d] %f\n", bRow, bCol, bSum);

	if (bCol == 0) {
		B[_id(bRow, cp1, cp2)] = ySum;
		printf("[[%d,%d] %f\n", bRow, 2, ySum);
	}
	//Summation done
}

__global__ void first_1(double *_x, double *_y, int cols, double* B)
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

__global__ void second(int cols, double* B, double* coeffs) {
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol = threadIdx.x;
	int bRow = threadIdx.y;

	//Gaussian elimination start
	// Pivotisation of the B matrix.
	double tmp;
	if (bRow == 0) {
		for (int i = 0; i < cp1; ++i)
			for (int k = i + 1; k < cp1; ++k) {
				if (B[_id(i, i, cp2)] < B[_id(k, i, cp2)]) {
					tmp = B[_id(i, bCol, cp2)];
					B[_id(i, bCol, cp2)] = B[_id(k, bCol, cp2)];
					B[_id(k, bCol, cp2)] = tmp;
				}
				__syncthreads();
			}
	}
	printf("--%d,%d] %f\n", bRow, bCol, B[_id(bRow, bCol, cp2)]);
	__syncthreads();

	if (bCol != cols + 1) {
		for (int i = 0; i < cols; ++i) {
			if (bRow > i )
				B[_id(bRow, bCol, cp2)] -= (B[_id(i, bCol, cp2)] * B[_id(bRow, i, cp2)]) / B[_id(i, i, cp2)];
			printf("%d+%d,%d] %f\n", i, bRow, bCol, B[_id(bRow, bCol, cp2)]);
			__syncthreads();
		}
	}
	printf("++%d,%d] %f\n", bRow, bCol, B[_id(bRow, bCol, cp2)]);
	__syncthreads();

	if (bRow != 0 || bCol != 0)
		return;

	for (int i = cols; i >= 0; --i) {
		double reduc = B[_id(i, cp1, cp2)];
		for (int j = i; j < cp1; ++j)
			reduc -= B[_id(i, j, cp2)] * coeffs[j];
		coeffs[i] = reduc / B[_id(i, i, cp2)];
	}

}


void kernelCall(double* _x, double* _y, double* _coeffs, int cols, double* B) {
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

	dim3 secondBlock(n+1, n);
	second << <1, secondBlock>> > (cols, B, _coeffs);

	cudaDeviceSynchronize();

	//LOOP_I(2)
	//	cudaStreamDestroy(stream[i]);
}