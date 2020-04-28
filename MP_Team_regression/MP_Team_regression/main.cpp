#include <stdio.h>
#include<vector>
#include<stdlib.h>
#include <numeric>
#include "DS_definitions.h"
#include "PolynomialRegression.h"

using namespace std;

#define f(x, c) c[0] + x*c[1] + x*x*c[2]

int main() {

	PolynomialRegression<double> a;

	vector<double> x(10);
	iota(x.begin(), x.end(), 0);
	vector<double> y{0,1,4,9,16,25,36,52,72,100 };
	int order = 2;
	vector<double> coeffs;

	a.fitIt(x, y, order, coeffs);

	for (double i : coeffs)
		printf("%f ",i);
	printf("\n");

	for (double i : x)
		printf("%.2f %.2f %f\n", i,y[i], f(i, coeffs));

	getchar();
	return 0;
}
