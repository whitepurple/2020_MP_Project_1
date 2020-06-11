#pragma once
#include "device_launch_parameters.h"

#define BLOCK_TID_1D	(threadIdx.x)
#define BLOCK_TID_2D	(blockDim.x * threadIdx.y + BLOCK_TID_1D)
#define BLOCK_TID_3D	(((blockDim.x * blockDim.y) * threadIdx.z)+ BLOCK_TID_2D)
#define NUM_THREAD_IN_BLOCK	(blockDim.x*blockDim.y*blockDim.z)
#define GRID_TID_1D	(blockIdx.x * NUM_THREAD_IN_BLOCK)
#define GRID_TID_2D	(blockIdx.y * (gridDim.x * NUM_THREAD_IN_BLOCK)) + GRID_TID_1D
#define GRID_TID_3D	(blockIdx.z * (gridDim.y * gridDim.x * NUM_THREAD_IN_BLOCK)) + GRID_TID_2D

#define TID_1D1D (GRID_TID_1D + BLOCK_TID_1D)