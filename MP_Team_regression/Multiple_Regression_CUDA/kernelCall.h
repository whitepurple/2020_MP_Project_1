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

#define numRows			 (197580)  //9880//
#define numRowsVerify	(1880)	// number of rows to use as a verifier
#define numRowsInput		(numRows - numRowsVerify)
#define numStats		(39)		// number of game stats from dataset

void kernelCall(double* _x, double* _y, int cols, double* B, int len, cudaStream_t stream);
void kernelCall2(double* _coeffs, int cols, double* B);