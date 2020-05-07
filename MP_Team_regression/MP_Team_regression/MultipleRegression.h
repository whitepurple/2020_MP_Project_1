#ifndef _MULTIPLE_REGRESSION_H
#define _MULTIPLE_REGRESSION_H  __MULTIPLE_REGRESSION_H
/*
*  y = a0 + a1 * A + a2 * B + ... + an * N

	Ba = Y
*/
#include<vector>
#include<stdlib.h>

template <class TYPE>
class MultipleRegression {
public:

	MultipleRegression();
	virtual ~MultipleRegression() {};

	bool fitIt(
		const std::vector<std::vector<TYPE>> & x,
		const std::vector<TYPE> & y,
		std::vector<TYPE> &     coeffs);
};

template <class TYPE>
MultipleRegression<TYPE>::MultipleRegression() {};

template <class TYPE>
bool MultipleRegression<TYPE>::fitIt(
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

	size_t N = x.size();		//입력 데이터 수
	int n = (int)(x[0].size());	//column 수
	int np1 = n + 1;			//n plus 1
	int np2 = n + 2;			//n plus 2

	std::vector<std::vector<TYPE> > X(np1, std::vector<TYPE>(np1, 0));
	// a = vector to store final coefficients.
	std::vector<TYPE> a(np1);
	// Y = vector to store values of sigma(xi * yi)
	std::vector<TYPE> Y(np1, 0);
	// B = normal augmented matrix that stores the equations.
	std::vector<std::vector<TYPE> > B(np1, std::vector<TYPE>(np2, 0));

	//0차, 1차 sigma (a_i, b_i, ...)
	X[0][0] = (double)N;
	for (int i = 1; i < np1; i++)
		for (int k = 0; k < N; ++k) 
			X[0][i] += (TYPE)x[k][i - 1];

	//2차 sigma (a_i * b_i ...)
	for (int i = 0; i < n; ++i) 
		for (int j = i; j < n; ++j) 
			for (int k = 0; k < N; ++k) 
					X[i+1][(j+1)] += (TYPE)(x[k][i] * x[k][j]);

	//계산된 X값들로 B행렬 생성
	for (int i = 0; i < np1; ++i) {
		for (int j = 0; j < np1; ++j) {
			B[i][j] = (i <= j) ? X[i][j]: X[j][i];
		}
	}

	//Y 벡터 생성
	for (int i = 0; i < np1; ++i) {
		for (int j = 0; j < N; ++j) {
			Y[i] += (TYPE)((i==0)?1:x[j][i-1])*y[j];
		}
	}


	// Load values of Y as last column of B
	for (int i = 0; i <= n; ++i)
		B[i][np1] = Y[i];

	n += 1;
	int nm1 = n - 1;

	// Pivotisation of the B matrix.
	for (int i = 0; i < n; ++i)
		for (int k = i + 1; k < n; ++k)
			if (B[i][i] < B[k][i])
				B[i].swap(B[k]);

	// Performs the Gaussian elimination.
	// (1) Make all elements below the pivot equals to zero
	//     or eliminate the variable.

	for (int i = 0; i < nm1; ++i)
		for (int k = i + 1; k < n; ++k) {
			//TYPE t = B[k][i] / B[i][i];
			for (int j = 0; j <= n; ++j)
				B[k][j] -= (B[i][j] * B[k][i]) / B[i][i];         // (1)
		}

	// Back substitution.
	// (1) Set the variable as the rhs of last equation
	// (2) Subtract all lhs values except the target coefficient.
	// (3) Divide rhs by coefficient of variable being calculated.
	for (int i = nm1; i >= 0; --i) {
		a[i] = B[i][n];							// (1)
		for (int j = 0; j < n; ++j)
			if (j != i)
				a[i] -= B[i][j] * a[j];			// (2)
		a[i] /= B[i][i];						// (3)
	}

	coeffs.resize(np1);					//계수 출력
	for (int i = 0; i < np1; ++i)
		coeffs[i] = a[i];

	return true;
}
#endif
