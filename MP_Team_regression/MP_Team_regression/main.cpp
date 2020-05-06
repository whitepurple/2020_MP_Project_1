#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <numeric>
#include "DS_definitions.h"
#include "PolynomialRegression.h"
#include "MultipleRegression.h"
#include "MultipleRegressionParallelized.h"
#include "CSVReader.h"
#include "OpenGL.h"

#define numRows 9880
#define numStats 11 // number of game stats from dataset

/* OpenGL */
std::vector<Point> dataInfo;
std::vector<std::vector<Point>> eachDataInfo;
std::vector<int> category;
int currMenuOper;

using namespace std;

// Simply calculating multi variable function
double calMF(vector<double> &x, vector<double> &coeff) {
	double ret = coeff[0];
	for (int i = 0; i < x.size(); ++i) {
		ret += x[i] * coeff[i + 1];
	}
	return ret;
}

// for input control
void pressAny() {
	printf("\nPress any key to continue...\n");
	int ch;
	while ((ch = getchar()) != '\n' && ch != EOF);
	getchar();
}

int main(int argc, char** argv) {

	MultipleRegression<double> mr;
	MultipleRegressionP<double> mrp;

	vector<double> coeffs(40, 0);		// coefficient
	vector<double> coeffsP(40, 0);		// coefficient for Polynomial Regression
	int selectX[numStats]{}, selectY{0};	// statArr to represent user selection

	// vector <int> x, y;		// x's for game stats. y's for blueWins

	// CSV Parsing from dataset.csv
	// CSVReader from open source project, dataset from kaggle data forum
	io::CSVReader<numStats> in("dataset.csv");		// 9880 rows, 40 columns in total. 4930 rows when blueWins == 1
	in.read_header(io::ignore_extra_column, "blueWins", "blueWardsPlaced", "blueWardsDestroyed", "blueTowersDestroyed",
		"blueKills", "blueDeaths", "blueAssists", "blueTotalExperience", "blueTotalMinionsKilled", "blueExperienceDiff", "blueGoldDiff");

	// User input
	string statStr[numStats]{ "blueWins", "blueWardsPlaced", "blueWardsDestroyed", "blueTowersDestroyed",
		"blueKills", "blueDeaths", "blueAssists", "blueTotalExperience", "blueTotalMinionsKilled", "blueExperienceDiff", "blueGoldDiff" };
	printf("[Available game stats for calculation]\n");
	for (int i = 0; i < numStats; ++i)
		printf("[%2d] %-23s\n", i + 1, statStr[i].c_str());
	printf("\nPlease Select stats by number to include on independent variables(=X) (input '0' to continue) : ");
	
	int inpt, inptCnt = 0;
	while (~scanf("%d", &inpt)) {
		if (inpt == 0) break;
		else if (inpt > 0 && inpt <= numStats) { selectX[inptCnt++] = inpt; }
		else printf("please select a proper number\n");
	}
	
	printf("Now, select ONE stats to be a range set(=Y) : ");
	while (!selectY && ~scanf("%d", &inpt)) {
		if (inpt > 0 && inpt <= numStats) { selectY = inpt; }
		else printf("please select a proper number\n");
	} pressAny();


	vector<vector<double>>	dataX{};	// dataX: game stats
	vector<double>			dataY{};	// dataY: blueWins itself
	// blueWins, blueWardsPlaced, blueWardsDestroyed, blueTowersDestroyed, blueKills, blueDeaths, blueAssists, blueTotalExperience, blueTotalMinionsKilled, blueExperienceDiff, blueGoldDiff;
	double stats[numStats]{};
	while (in.read_row(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10])) {
		// Setting independent variables set and result set
		vector<double> tmp;
		for (int i = 0; i < inptCnt; ++i)
			tmp.push_back(stats[selectX[i] - 1]);
		
		dataX.push_back(tmp);
		dataY.push_back(stats[selectY - 1]);
	}

	// Input Data Visualization
	inputData(dataX, dataY);
	getCategory(selectX, inptCnt);
	initGL(&argc, argv);

	// Multiple Regression
	printf("\n[ Multiple Regression on Progress... ]\n");
	mr.fitIt(dataX, dataY, coeffs);
	printf("Completed function: f(x) = %f", coeffs[0]);
	for (int i = 0; i < inptCnt; ++i)
		printf(" + %f * x%d", coeffs[i + 1], i);
	printf("\n");

	// Multiple Regression Parallelized
	printf("\n[ Parallelized Multiple Regression on Progress... ]\n");
	mrp.fitIt(dataX, dataY, coeffsP);
	printf("Completed function: f(x) = %f", coeffsP[0]);
	for (int i = 0; i < inptCnt; ++i)
		printf(" + %f * x%d", coeffsP[i + 1], i);
	printf("\n\n");

	// Verifying results with real dataset
	printf("\n\n[ Verifying with Dataset ]\n"); 
	printf("How many dataset do you want to verify? : ");
	while (~scanf("%d", &inpt)) {
		if (inpt > 0 && inpt < numRows) break;
		else if (inpt >= numRows) printf("I'm afraid there's only 9880 rows in dataset...\n");
		else printf("please select a proper number\n");
	} pressAny();

	printf("\nGiven Factors were \nX { ");
	for(int i = 0; i < inptCnt; ++i)
		printf("%s, ", statStr[selectX[i] - 1].c_str());
	printf("}, \nY { %s }\n\n", statStr[selectY - 1].c_str());

	for (int i = 0; i < inpt; ++i) {
		printf("X[%3d]{", i);
		for (double d : dataX[i])
			printf("%6.2f ", d);
		printf("}");

		printf(" | Y[%3d](%8.2f) | f(x)=%12.3f (p)f(x)=%12.3f\n", i, dataY[i], calMF(dataX[i], coeffs), calMF(dataX[i], coeffsP));
	}

	pressAny();
	return 0;
}
