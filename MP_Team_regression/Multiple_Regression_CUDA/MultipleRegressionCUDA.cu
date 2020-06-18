#include "kernelCall.h"

#define NUM_T_IN_BLOCK 512

__global__ void first(double *_x, double *_y, int cols, double* B)
{
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol  = threadIdx.x;
	int bRow = threadIdx.y;

	double bSum = 0;
	double ySum = 0;
	double x1, x2;

	int bRowm1 = bRow - 1;
	int bColm1 = bCol - 1;

	for (int i = 0; i < numRowsInput; i++) {
		////make B
		x1 = (bRow == 0) ? 1 : _x[i * cols + bRowm1];
		x2 = (bCol == 0) ? 1 : _x[i * cols + bColm1];
		bSum += x1 * x2;

		////make Y
		ySum += x1 * _y[i];
	}
	
	B[_id(bRow, bCol, cp2)] = bSum;
	//printf("[[%d,%d] %f\n", bRow, bCol, bSum);

	if (bCol == 0) {
		B[_id(bRow, cp1, cp2)] = ySum;
		//printf("[[%d,%d] %f\n", bRow, cp1, ySum);
	}
	//Summation done
}

__device__ void warpReduce(volatile double* _localVal, int _tid)
{
	_localVal[_tid] += _localVal[_tid + 32];
	_localVal[_tid] += _localVal[_tid + 16];
	_localVal[_tid] += _localVal[_tid + 8];
	_localVal[_tid] += _localVal[_tid + 4];
	_localVal[_tid] += _localVal[_tid + 2];
	_localVal[_tid] += _localVal[_tid + 1];
}

__global__ void first_1(double *_x, double *_y, int cols, double* B, int _len)
{
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol = blockIdx.x;
	int bRow = blockIdx.y;

	int tID = blockIdx.z*blockDim.x + threadIdx.x;

	volatile __shared__ double local[NUM_T_IN_BLOCK];
	local[BLOCK_TID_1D] = 0;

	if (tID >= _len)
		return;

	double x1, x2;

	////make B
	x1 = ((bRow == 0) ? 1 : _x[tID * cols + bRow - 1]);
	x2 = ((bCol == 0) ? 1 : ((bCol == cp1) ? _y[tID] : _x[tID * cols + bCol - 1]));
	local[BLOCK_TID_1D] = x1 * x2;

	__syncthreads();

	//if (BLOCK_TID_1D < 512) {
	//	local[BLOCK_TID_1D] += local[BLOCK_TID_1D + 512];
	//}
	//__syncthreads();

	if (BLOCK_TID_1D < 256) {
		local[BLOCK_TID_1D] += local[BLOCK_TID_1D + 256];
	}
	__syncthreads();

	if (BLOCK_TID_1D < 128) {
		local[BLOCK_TID_1D] += local[BLOCK_TID_1D + 128];
	}
	__syncthreads();

	if (BLOCK_TID_1D < 64) {
		local[BLOCK_TID_1D] += local[BLOCK_TID_1D + 64];
	}
	__syncthreads();

	if (BLOCK_TID_1D < 32) {
		warpReduce(local, BLOCK_TID_1D);
	}

	if (threadIdx.x == 0) {
		atomicAdd(&B[_id(bRow, bCol, cp2)], local[0]);
		//printf("[[%d,%d] %f\n", bRow, bCol, local[0]);
	}
	//Summation done
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
			}
	}
	//printf("--%d,%d] %f\n", bRow, bCol, B[_id(bRow, bCol, cp2)]);
	__syncthreads();

	for (int i = 0; i < cols; ++i) {
		if (bRow > i) {
			for (int j = 0; j < cp1; ++j) {
				if (bCol == 0)
					B[_id(bRow, j, cp2)] -= (B[_id(i, j, cp2)] * B[_id(bRow, i, cp2)]) / B[_id(i, i, cp2)];
			}
		}
		__syncthreads();
	}
	//printf("++%d,%d] %f\n", bRow, bCol, B[_id(bRow, bCol, cp2)]);
	__syncthreads();

	if (bRow != 0 || bCol != 0)
		return;

	for (int i = cols; i >= 0; --i) {
		double reduc = B[_id(i, cp1, cp2)];
		for (int j = i; j < cp1; ++j)
			reduc -= ( B[_id(i, j, cp2)]* coeffs[j]);
		coeffs[i] = (reduc / B[_id(i, i, cp2)]);
	}
}

//__global__ void first_2(double* B)

void kernelCall(double* _x, double* _y, int cols, double* B, int len, cudaStream_t stream) {
	int n = cols + 1;
	int height = ceil((float)len / NUM_T_IN_BLOCK);
	dim3 first_1Grid(n+1, n, height);
	//height = ceil((float)height / NUM_T_IN_BLOCK);

	first_1<< <first_1Grid, NUM_T_IN_BLOCK, 0, stream>> > (_x, _y, cols, B, len);
	//first << <1, firstBlock >> > (_x, _y, cols, B);
}

void kernelCall2(double* _coeffs, int cols, double* B) {
	int n = cols + 1;
	dim3 secondBlock(n + 1, n);
	second << <1, secondBlock >> > (cols, B, _coeffs);
}