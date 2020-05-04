#include <iostream>
#include <vector>
#include <numeric>
#include "DS_definitions.h"
#include "PolynomialRegression.h"
#include "MultipleRegression.h"
#include "MultipleRegressionParallelized.h"

/* OpenGL */
#include "OpenGL.h"
std::vector<Point> dataInfo;

using namespace std;

#define pf(x, c) c[0] + x*c[1] + x*x*c[2] +  x*x*x*c[3]
#define mf(x,c) c[0] + x[0]*c[1] + x[1]*c[2] + x[2]*c[3]

int main(int argc, char** argv) {
	/* Data */
	vector<double> x{ 1.47, 1.50, 1.52, 1.55, 1.57, 1.60, 1.63, 1.65, 1.68, 1.70, 1.73, 1.75, 1.78, 1.80, 1.83 };
	vector<double> y{ 52.21, 53.12, 54.48, 55.84, 57.20, 58.57, 59.93, 61.29, 63.11, 64.47, 66.28, 68.10, 69.92, 72.19, 74.46 };
	
	/* OpenGL Settings */
	inputData(x, y);
	initGL(&argc, argv);

	/* Regression Calculation */
	PolynomialRegression<double> pr;
	MultipleRegression<double> mr;
	MultipleRegressionP<double> mrp;

	vector<vector<double>> xx;
	int order = 3;
	vector<double> coeffs(order,0);
	vector<double> coeffsP(order,0);
	

	for (int i = 0; i < 15;i++) {
		double xi = x[i];
		vector<double> elem{xi, xi*xi, xi*xi*xi};	//���� ȸ�� �׽�Ʈ�� ���� ����ȸ�� �Է�
		xx.push_back(elem);
	}

	//pr.fitIt(x, y, order, coeffs);	//���� ȸ��
	mr.fitIt(xx, y, coeffs);			//���� ȸ��

	for (double i : coeffs)				//����׺��� ���
		printf("%f ",i);
	printf("\n");

	mrp.fitIt(xx, y, coeffsP);			//���� ȸ��

	for (double i : coeffsP)			//����׺��� ���
		printf("%f ", i);
	printf("\n");

	//�Է°��� ���İ� ��
	for (int i = 0; i < 11; ++i) {
		printf("x(");
		for (double d : xx[i])
			printf("%.2f ", d);
		printf(") ");

		printf("y(%.2f) f(x)=%f (p)f(x)=%f\n", y[i], mf(xx[i], coeffs), mf(xx[i], coeffsP));

	}
	getchar();
	return 0;
}
