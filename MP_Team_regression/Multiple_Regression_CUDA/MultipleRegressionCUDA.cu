#include "kernelCall.h"

#define NUM_T_IN_BLOCK 512

__global__ void basefirst(double *_x, double *_y, int cols, double* B)
{
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol = threadIdx.x;
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

__global__ void first_v1(double *_x, double *_y, int cols, double* B, int _len)
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

// Row Segment Atomic Version
__global__ void first_segment_atomic(double *_x, double *_y, int cols, double* B)
{
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol = threadIdx.x;
	int bRow = threadIdx.y;

	double bSum = 0;
	double ySum = 0;
	double x1, x2;

	int bRowm1 = bRow - 1;
	int bColm1 = bCol - 1;


	// if use shared memory
	// size = 8bytes * cols * size < 64KB

	int size = (numRowsInput / gridSize) + 1;
	for (int i = size * blockIdx.x; i < size * (blockIdx.x + 1); i++) {

		// ������ �Ѿ �� �����Ƿ�
		if (i < numRowsInput) {
			////make B
			x1 = (bRow == 0) ? 1 : _x[i * cols + bRowm1];
			x2 = (bCol == 0) ? 1 : _x[i * cols + bColm1];
			bSum += x1 * x2;

			////make Y
			ySum += x1 * _y[i];
		}

	}

	//printf("bSum : %lf\n", bSum);

	atomicAdd(&B[_id(bRow, bCol, cp2)], bSum);
	//printf("[[%d,%d] %f\n", bRow, bCol, bSum);

	if (bCol == 0) {
		atomicAdd(&B[_id(bRow, cp1, cp2)], ySum);
		//printf("[[%d,%d] %f\n", bRow, cp1, ySum);
	}
	//Summation done

}

// Row Segment Reduction Version
__global__ void first_segment_reduction(double *_x, double *_y, int cols, double* res, int len) {
	// Base Variable
	int cp1 = cols + 1;
	int cp2 = cols + 2;
	int bCol = threadIdx.x;
	int bRow = threadIdx.y;

	double bSum = 0;
	double ySum = 0;
	double x1, x2;

	int bRowm1 = bRow - 1;
	int bColm1 = bCol - 1;


	// ���� �۾� ���� - gridSize = block ����
	// ��ü ������ 200000���� ���� �� block ������ ���� �̸� block �� �Ҵ� ũ�� ����
	// ������ �Ҽ����� ���� �� �����Ƿ� Ceiling�� ���� 1 ����
	int size = (len / gridSize) + 1;
	int i = 0;

	for (i = size * blockIdx.x; i < size * (blockIdx.x + 1); i++) {
		// ������ �Ѿ �� �����Ƿ�
		if (i >= len) break;

		////make B
		x1 = (bRow == 0) ? 1 : _x[i * cols + bRowm1];
		x2 = (bCol == 0) ? 1 : _x[i * cols + bColm1];
		bSum += x1 * x2;

		////make Y
		ySum += x1 * _y[i];
	}

	__syncthreads();

	// Global Memory�� res�� ���� ����
	// ���� threadIdx.x, threadIdx.y ���� ���������� �����ϴ� �ε���
	// ���� B�� ���� ������ ������ bSum ���� ���������� ��ġ
	// ���� �� block�� ���� ������ ��ȣ���� ���������� �����ϹǷ� blockIdx.x�� ������ �ȴ�.
	res[blockIdx.x + _id(bRow, bCol, cp2) * gridSize] += bSum;

	if (bCol == 0) {
		res[blockIdx.x + _id(bRow, cp1, cp2) * gridSize] += ySum;
	}
	//Summation done
}

__global__ void reduction(double* B, double* res) {
	// �ݵ�� gridSize�� 1024���� �۾ƾ� �� - blockSize�� �ִ밡 1024
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	// ���⼭ gridSize�� reduction kernel�� blockSize
	__shared__ double local[gridSize]; // if gridSize : 1024, shared memory : 8KB

	local[threadIdx.x] = res[tid];
	__syncthreads();
	
	// Block���� Reduction ����
	int offset = 1;
	while (offset < gridSize) {
		if (threadIdx.x % (2 * offset) == 0) {
			local[threadIdx.x] += local[threadIdx.x + offset];
		}

		__syncthreads();

		offset *= 2;
	}
	
	// Res�� blockIdx.x ���� �й�Ǿ� �����Ƿ� 
	// Reduction ��� ���� ��� �� block�� 0�� thread�� ����
	// B�� blockIdx.x�� �������� ���ʴ�� �־��ָ� �ȴ�.
	if (threadIdx.x == 0) {
		B[blockIdx.x] = local[threadIdx.x];
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
			reduc -= (B[_id(i, j, cp2)] * coeffs[j]);
		coeffs[i] = (reduc / B[_id(i, i, cp2)]);
	}
}

//__global__ void first_2(double* B)

void kernelCall_v1_first(double* _x, double* _y, int cols, double* B, int len, cudaStream_t stream) {
	int n = cols + 1;
	int height = ceil((float)len / NUM_T_IN_BLOCK);
	dim3 first_1Grid(n+1, n, height);
	//height = ceil((float)height / NUM_T_IN_BLOCK);

	first_v1<< <first_1Grid, NUM_T_IN_BLOCK, 0, stream>> > (_x, _y, cols, B, len);
	//first << <1, firstBlock >> > (_x, _y, cols, B);
}

void kernelCall_second(double* _coeffs, int cols, double* B) {
	int n = cols + 1;
	dim3 secondBlock(n + 1, n);
	second << <1, secondBlock >> > (cols, B, _coeffs);
}

void kernelCall_Segment(double* _x, double* _y, int cols, int len, double* res, cudaStream_t stream) {
	int n = cols + 1;
	dim3 firstBlock(n, n);
	first_segment_reduction << <gridSize, firstBlock, 0, stream >> > (_x, _y, cols, res, len);
}

void kernelCall_Reduction(int cols, double* B, double* res) {
	int n = cols + 1;
	reduction << <n * (n + 1), gridSize >> > (B, res);
}

void kernelCall_Debug(double* _x, double* _y, int cols, double* B, int len, double* res, cudaStream_t stream) {
	//int n = cols + 1;

	// res�� 1111111 2222222 333333 �� ���� ������ ����

	//dim3 firstBlock(n, n);
	
	// ��� Version
	/*
	int height = ceil((float)len / NUM_T_IN_BLOCK);
	dim3 first_1Grid(n, n, height);
	first_1 << <first_1Grid, NUM_T_IN_BLOCK >> > (_x, _y, cols, B, len);
	*/

	//printf("Rows : %d\n", len);

	
	// Reduction Version
	//first_segment_reduction <<<gridSize, firstBlock, 0, stream>>> (_x, _y, cols, res, len);
	// Implicit Synchronize
	//reduction <<<n * (n + 1), gridSize >>> (B, res);	

	/*
	// Basic
	int height = ceil((float)len / NUM_T_IN_BLOCK);
	dim3 first_1Grid(n, n, height);
	//timer.onTimer(1);
	first_1 <<<first_1Grid, NUM_T_IN_BLOCK >> > (_x, _y, cols, mB, len);
	*/

	// Atomic Version
	//first_segment_atomic << <gridSize, firstBlock >> > (_x, _y, cols, B);

	
	// Debug ��
	/*
	double *tmpB;
	tmpB = (double*)malloc(sizeof(double) * n * (n + 1) * gridSize);
	memset(tmpB, 0, sizeof(double) * n * (n + 1) * gridSize);

	double *tmpB2;
	tmpB2 = (double*)malloc(sizeof(double) * n * (n + 1) * gridSize);
	memset(tmpB2, 0, sizeof(double) * n * (n + 1) * gridSize);
	
	cudaMemcpy(tmpB, B, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);
	cudaMemcpy(tmpB2, mB, sizeof(double) * n * (n + 1), cudaMemcpyDeviceToHost);

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n + 1; j++) {
			if (tmpB[i * n + j] != tmpB2[i * n + j]) {
				printf("id : %d, B : %lf, mB : %lf\n", i * n + j, tmpB[i * n + j], tmpB2[i * n + j]);
			}
		}
	}
	*/

}