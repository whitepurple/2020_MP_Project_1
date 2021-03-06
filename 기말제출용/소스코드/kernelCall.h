#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DS_timer.h"
#include "DS_definitions.h"
#include "indexing.cuh"

#define NUM_BLOCK 256

#define numRows			(197580)					// 9880//
#define numRowsVerify	(1880)						// number of rows to use as a verifier
#define numRowsInput	(numRows - numRowsVerify)	// number of rows to use as train
#define numStats		(39)						// number of game stats from dataset
#define gridSize		(64)						// Row Segment Count

void kernelCall_v1_first(double* _x, double* _y, int cols, double* B, int len, cudaStream_t stream);
void kernelCall_second(double* _coeffs, int cols, double* B);

//////////
void kernelCall_v2_Segment(double* _x, double* _y, int cols, int len, double *res, cudaStream_t stream);
void kernelCall_v2_Reduction(int cols, double* B, double* res);
void kernelCall_v2_Debug(double* _x, double* _y, int cols, double* B, int len, double *res, cudaStream_t stream);
