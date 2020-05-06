#ifndef _MULTIPLE_REGRESSION_P_H
#define _MULTIPLE_REGRESSION_P_H  __MULTIPLE_REGRESSION_P_H
/*
*  y = a0 + a1 * A + a2 * B + ... + an * N

	Ba = Y
*/
#include<vector>
#include<stdlib.h>
#include <omp.h>

#define cache (int)32
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
	#pragma omp parallel num_threads(NUMTHREADS)
	{
		#pragma omp for schedule(dynamic) nowait
		for (int i = 1; i < np1; i++)
			for (int k = 0; k < N; ++k)
				X[0][i*cache] += (TYPE)x[k][i - 1];

		//2차 sigma
		#pragma omp for schedule(dynamic) nowait
		for (int i = 0; i < n; ++i)
			for (int j = i; j < n; ++j)
				for (int k = 0; k < N; ++k)
					X[i + 1][(j + 1)*cache] += (TYPE)(x[k][i] * x[k][j]);

		#pragma omp for schedule(dynamic) nowait
		for (int i = 0; i < np1; ++i) {
			for (int j = 0; j < N; ++j) {
				Y[i*cache] += (TYPE)((i == 0) ? 1 : x[j][i - 1])*y[j];
			}
		}

	}

	#pragma omp parallel num_threads(NUMTHREADS)
	{
		#pragma omp for schedule(dynamic) nowait
		for (int i = 0; i < np1; ++i) {
			for (int j = 0; j < np1; ++j) {
				B[i][j*cache] = (i <= j) ? X[i][j*cache] : X[j][i*cache];
			}
		}

		// Load values of Y as last column of B
		#pragma omp for schedule(dynamic) nowait
		for (int i = 0; i <= n; ++i)
			B[i][np1*cache] = Y[i*cache];
	}

	n += 1;
	int nm1 = n - 1;

	// Pivotisation of the B matrix.	//유사버블정렬
	for (int i = 0; i < n; ++i) {
		int first = i % 2;
		#pragma omp parallel for default(none),shared(B,first)
		for (int k = first; k < n-1; k+=2)
			if (B[k][i*cache] < B[k+1][i*cache])
				B[k].swap(B[k+1]);
	}

	// Performs the Gaussian elimination.
	// (1) Make all elements below the pivot equals to zero
	//     or eliminate the variable.
	for (int i = 0; i < nm1; ++i)
		#pragma omp parallel for schedule(dynamic)
		for (int k = i + 1; k < n; ++k) {
			TYPE t = B[k][i*cache] / B[i][i*cache];
			for (int j = 0; j <= n; ++j)
				B[k][j*cache] -= t * B[i][j*cache];         // (1)
		}
	// Back substitution.
	// (1) Set the variable as the rhs of last equation
	// (2) Subtract all lhs values except the target coefficient.
	// (3) Divide rhs by coefficient of variable being calculated.

	for (int i = nm1; i >= 0; --i) {
		TYPE reduc = B[i][n*cache];                   // (1)
		#pragma omp parallel for reduction(-:reduc) schedule(dynamic)
		for (int j = 0; j < n; ++j)
			if (j != i)
				reduc -= B[i][j*cache] * a[j*cache];       // (2)
		a[i*cache] = reduc / B[i][i*cache];		// (3)
	}

	coeffs.resize(np1);		//계수 출력
	for (size_t i = 0; i < np1; ++i)
		coeffs[i] = a[i*cache];

	return true;
}
#endif
