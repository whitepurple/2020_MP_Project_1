#ifndef _MULTIPLE_REGRESSION_P_H
#define _MULTIPLE_REGRESSION_P_H  __MULTIPLE_REGRESSION_P_H
/*
*  y = a0 + a1 * A + a2 * B + ... + an * N

	Ba = Y
*/
#include<vector>
#include<stdlib.h>
#include <omp.h>

#include "DS_timer.h"

#define cache 8
#define NUMTHREADS 8

template <class TYPE>
class MultipleRegressionP {
public:

	MultipleRegressionP();
	virtual ~MultipleRegressionP() {};

	bool fitIt(
		const std::vector<std::vector<TYPE>> & x,
		const std::vector<TYPE> & y,
		std::vector<TYPE> &     coeffs);
};

template <class TYPE>
MultipleRegressionP<TYPE>::MultipleRegressionP() {};

template <class TYPE>
bool MultipleRegressionP<TYPE>::fitIt(
	const std::vector<std::vector<TYPE>> & x,
	const std::vector<TYPE> & y,
	std::vector<TYPE> &       coeffs)
{
	DS_timer timer(7);
	timer.setTimerName(0, (char*)"test0");
	timer.setTimerName(1, (char*)"test1");
	timer.setTimerName(2, (char*)"test2");
	timer.setTimerName(3, (char*)"test3");
	timer.setTimerName(4, (char*)"test4");
	timer.setTimerName(5, (char*)"test5");
	timer.setTimerName(6, (char*)"test6");


	// The size of xValues and yValues should be same
	if (x.size() != y.size()) {
		throw std::runtime_error("The size of x & y arrays are different");
		return false;
	}
	// The size of xValues and yValues cannot be 0, should not happen
	if (x.size() == 0 || y.size() == 0) {
		throw std::runtime_error("The size of x or y arrays is 0");
		return false;
	}

	size_t N = x.size();	//입력 데이터 수
	int n = (int)(x[0].size());	//column 수
	int np1 = n + 1;		//n plus 1
	int np2 = n + 2;	//n plus 2

	std::vector<std::vector<TYPE> > X(np1, std::vector<TYPE>(np1*cache, 0));
	// a = vector to store final coefficients.
	std::vector<TYPE> a(np1*cache);
	// Y = vector to store values of sigma(xi * yi)
	std::vector<TYPE> Y(np1*cache, 0);
	// B = normal augmented matrix that stores the equations.
	std::vector<std::vector<TYPE> > B(np1, std::vector<TYPE>(np2*cache, 0));

	//0차, 1차 sigma
	X[0][0] = (TYPE)N;

	TYPE t = 0;

	timer.onTimer(0);
	for (int i = 1; i < np1; i++) {
		t = 0;
		#pragma omp parallel for reduction(+: t)
		for (int k = 0; k < N; ++k) {
			t += (TYPE)x[k][i - 1];
		}
		X[0][i * cache] = t;
	}
	timer.offTimer(0);

	timer.onTimer(1);
	for (int i = 0; i < np1; ++i) {
		t = 0;
		#pragma omp parallel for reduction(+: t)
		for (int k = 0; k < N; ++k) {
			t += (TYPE)((i == 0) ? 1 : x[k][i - 1]) * y[k];
		}
		Y[i * cache] = t;
	}
	timer.offTimer(1);

	//2차 sigma
	TYPE t2, t3, t4, t5, t6, t7, t8;
	timer.onTimer(2);
	for (int i = 1; i < np1; i++) {
		for (int j = i; j < np1; j+=8) {
			t = t2 = t3 = t4 = t5 = t6 = t7 = t8 = 0;
			#pragma omp parallel for reduction(+: t, t2, t3, t4, t5, t6, t7, t8)
			for (int k = 0; k < N; k++) {
				t += (TYPE)(x[k][i - 1] * x[k][j - 1]);
				if (j + 1 < np1) t2 += (TYPE)(x[k][i - 1] * x[k][j]);
				if (j + 2 < np1) t3 += (TYPE)(x[k][i - 1] * x[k][j + 1]);
				if (j + 3 < np1) t4 += (TYPE)(x[k][i - 1] * x[k][j + 2]);
				if (j + 4 < np1) t5 += (TYPE)(x[k][i - 1] * x[k][j + 3]);
				if (j + 5 < np1) t6 += (TYPE)(x[k][i - 1] * x[k][j + 4]);
				if (j + 6 < np1) t7 += (TYPE)(x[k][i - 1] * x[k][j + 5]);
				if (j + 7 < np1) t8 += (TYPE)(x[k][i - 1] * x[k][j + 6]);
			}
			X[i][j * cache] = t;
			if (j + 1 < np1) X[i][(j + 1) * cache] = t2;
			if (j + 2 < np1) X[i][(j + 2) * cache] = t3;
			if (j + 3 < np1) X[i][(j + 3) * cache] = t4;
			if (j + 4 < np1) X[i][(j + 4) * cache] = t5;
			if (j + 5 < np1) X[i][(j + 5) * cache] = t6;
			if (j + 6 < np1) X[i][(j + 6) * cache] = t7;
			if (j + 7 < np1) X[i][(j + 7) * cache] = t8;
		}
	}
	timer.offTimer(2);

	timer.onTimer(3);
	#pragma omp parallel for schedule(dynamic)
	for (int i = 0; i < np1; ++i) {
		for (int j = 0; j < np1; ++j) {
			B[i][j * cache] = (i <= j) ? X[i][j * cache] : X[j][i * cache];
		}
		// Load values of Y as last column of B
		B[i][np1 * cache] = Y[i * cache];
	}
	timer.offTimer(3);

	n += 1;
	int nm1 = n - 1;

	TYPE* tmp = NULL;

	timer.onTimer(4);
	//병렬화 불가
	// Pivotisation of the B matrix.
	for (int i = 0; i < n; ++i)
		for (int k = i + 1; k < n; ++k)
			if (B[i][i * cache] < B[k][i * cache]) {
				B[i].swap(B[k]);
			}
	timer.offTimer(4);
	
	// Performs the Gaussian elimination.
	// (1) Make all elements below the pivot equals to zero
	//     or eliminate the variable.
	timer.onTimer(5);
	for (int i = 0; i < nm1; ++i) {
		TYPE bii = B[i][i * cache];
		#pragma omp parallel for
		for (int k = i + 1; k < n; ++k) {
			for (int j = 0; j < np1; ++j) {
				B[k][j * cache] -= (B[i][j * cache] * B[k][i * cache]) / bii;         // (1)
			}
		}
	}
	timer.offTimer(5);
	// Back substitution.
	// (1) Set the variable as the rhs of last equation
	// (2) Subtract all lhs values except the target coefficient.
	// (3) Divide rhs by coefficient of variable being calculated.

	timer.onTimer(6);
	for (int i = nm1; i >= 0; --i) {
		TYPE reduc = B[i][n*cache];                   // (1)
		#pragma omp parallel for reduction(-:reduc) 
		for (int j = 0; j < n; ++j)
			if (j != i)
				reduc -= B[i][j*cache] * a[j*cache];       // (2)
		a[i*cache] = reduc / B[i][i*cache];		// (3)
	}
	timer.offTimer(6);
	timer.printTimer();

	coeffs.resize(np1);		//계수 출력
	#pragma omp parallel for
	for (int i = 0; i < np1; ++i)
		coeffs[i] = a[i*cache];

	return true;
}
#endif
