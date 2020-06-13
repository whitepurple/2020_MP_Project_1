#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <numeric>
#include "PolynomialRegression.h"
#include "CSVReader.h"
#include "OpenGL.h"
#include "DS_definitions.h"
#include "DS_timer.h"
#include "MultipleRegression.h"
#include "MultipleRegressionParallelized.h"
#include "kernelCall.h"

#define ABS(X) ((X) < 0 ? -(X) : (X))
#define EPSILON 0.000001

#define numRows			 (197580)  //9880//
#define numRowsVerify	(1880)	// number of rows to use as a verifier
#define numRowsInput		(numRows - numRowsVerify)
#define numStats		(39)		// number of game stats from dataset
#define testnum			(400)

/* OpenGL */
std::vector<Point> dataInfo;						// save data information for OpenGL
std::vector<std::vector<Point>> eachDataInfo;		// save data when user choose to show each of data visualization
std::vector<int> category;							// category of X-data
int yStats;											// category of Y-data
int currMenuOper;									// menu operation number

// Simply calculating multi variable function
double calMF(std::vector<double> &x, std::vector<double> &coeff) {
	double ret = coeff[0];
	for (int i = 0; i < x.size(); ++i) {
		ret += x[i] * coeff[i + 1];
	}
	return ret;
}

double calMF(std::vector<double> &x, double *coeff) {
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
	DS_timer timer(3);
	timer.setTimerName(0, (char*)"Serial");
	timer.setTimerName(1, (char*)"Parallel OpenMP");
	timer.setTimerName(2, (char*)"Parallel CUDA");

	srand(time(NULL));

	MultipleRegression<double> mr;
	MultipleRegressionP<double> mrp;

	std::vector<double> coeffs(numStats+1, 0);		// coefficient
	std::vector<double> coeffsP(numStats+1, 0);		// coefficient for Polynomial Regression

	int selectX[numStats]{}, selectY{0};	// statArr to represent user selection

	// vector <int> x, y;		// x's for game stats. y's for blueWins

	// CSV Parsing from dataset.csv
	// CSVReader from open source project, dataset from kaggle data forum
	io::CSVReader<numStats> in("dataset.csv");		// 9880 rows, 40 columns in total. 4930 rows when blueWins == 1
	in.read_header(io::ignore_extra_column,
		"blueWins",
		"blueWardsPlaced", "blueWardsDestroyed", "blueFirstBlood", "blueKills", "blueDeaths", "blueAssists",
		"blueEliteMonsters", "blueDragons", "blueHeralds", "blueTowersDestroyed", "blueTotalGold", "blueAvgLevel",
		"blueTotalExperience", "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled", "blueGoldDiff", "blueExperienceDiff",
		"blueCSPerMin", "blueGoldPerMin",
		"redWardsPlaced", "redWardsDestroyed", "redFirstBlood", "redKills", "redDeaths", "redAssists",
		"redEliteMonsters", "redDragons", "redHeralds", "redTowersDestroyed", "redTotalGold", "redAvgLevel",
		"redTotalExperience", "redTotalMinionsKilled", "redTotalJungleMinionsKilled", "redGoldDiff", "redExperienceDiff",
		"redCSPerMin", "redGoldPerMin");

	// User input
	std::string statStr[numStats]{ "blueWins",
		"blueWardsPlaced", "blueWardsDestroyed", "blueFirstBlood", "blueKills", "blueDeaths", "blueAssists",
		"blueEliteMonsters", "blueDragons", "blueHeralds", "blueTowersDestroyed", "blueTotalGold", "blueAvgLevel",
		"blueTotalExperience", "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled", "blueGoldDiff", "blueExperienceDiff",
		"blueCSPerMin", "blueGoldPerMin",
		"redWardsPlaced", "redWardsDestroyed", "redFirstBlood", "redKills", "redDeaths", "redAssists",
		"redEliteMonsters", "redDragons", "redHeralds", "redTowersDestroyed", "redTotalGold", "redAvgLevel",
		"redTotalExperience", "redTotalMinionsKilled", "redTotalJungleMinionsKilled", "redGoldDiff", "redExperienceDiff",
		"redCSPerMin", "redGoldPerMin" };
	printf("[Available game stats for calculation]\n");
	printf("[%2d] %-28s\n", 1, statStr[0].c_str());
	for (int i = 1; i < numStats/2+1; ++i)
		printf("[%2d] %-28s  [%2d] %-28s\n", i + 1, statStr[i].c_str(), numStats / 2+i + 1, statStr[numStats / 2 + i].c_str());
	printf("[100] %-28s\n", "test");
	printf("\nPlease Select stats by number to include on independent variables(=X) (input '0' to continue) : ");
	
	int inpt, inptCnt = 0;
	while (~scanf("%d", &inpt)) {
		if (inpt == 100) {	//최대 열 16개에 대하여 테스트
			for (int i = 0; i < 16; i++) selectX[i] = i + 10;
			inptCnt += 16;
			break;
		}
		else if (inpt == 0) break;
		else if (inpt > 0 && inpt <= numStats) { selectX[inptCnt++] = inpt; }
		else printf("please select a proper number\n");
	}
	
	printf("Now, select ONE stats to be a range set(=Y) : ");
	while (!selectY && ~scanf("%d", &inpt)) {
		if (inpt > 0 && inpt <= numStats) { selectY = inpt; }
		else printf("please select a proper number\n");
	} pressAny();


	
	std::vector<std::vector<double>>	dataX{}, verifyX{};	// dataX: game stats as X	verifyX: X factor data for verifying
	std::vector<double>			dataY{}, verifyY{};	// dataY: Y factor itself	verifyY: ,,, data for verifying

	///////////////////// memery for CUDA

	double coeffsP_2[numStats]{ 0 };

	double* dcoeffsP_2;		// coefficient for Polynomial Regression
	cudaMalloc(&dcoeffsP_2, sizeof(double)*numStats);
	cudaMemset(dcoeffsP_2, 0, sizeof(double)*numStats);

	double *hx, *hy, *dx, *dy, *matB[numStats + 1];

	cudaMallocHost(&hx, sizeof(double)*numStats*numRows);
	memset(hx, 0, sizeof(double)*numStats*numRows);
	cudaMallocHost(&hy, sizeof(double)*numStats*numRows);
	memset(hy, 0, sizeof(double)*numStats*numRows);

	
	cudaMalloc(&dx, sizeof(double)*numStats*numRows);
	cudaMemset(dx, 0, sizeof(double)*numStats*numRows);
	cudaMalloc(&dy, sizeof(double)*numStats*numRows);
	cudaMemset(dy, 0, sizeof(double)*numStats*numRows);

	for (int i = 0; i < numStats + 2; i++) {
		cudaMalloc(&matB[i], sizeof(double)*(numStats+2));
		cudaMemset(matB[i], 0, sizeof(double)*(numStats+2));
	}

	/////////////////////
	int counter = 0;
	// blueWins, blueWardsPlaced, blueWardsDestroyed, blueTowersDestroyed, blueKills, blueDeaths, blueAssists, blueTotalExperience, blueTotalMinionsKilled, blueExperienceDiff, blueGoldDiff;
	double stats[numStats]{};
	while (in.read_row(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6], stats[7], stats[8], stats[9],
		stats[10], stats[11], stats[12], stats[13], stats[14], stats[15], stats[16], stats[17], stats[18], stats[19], 
		stats[20], stats[21], stats[22], stats[23], stats[24], stats[25], stats[26], stats[27], stats[28], stats[29],
		stats[30], stats[31], stats[32], stats[33], stats[34], stats[35], stats[36], stats[37], stats[38] ) && counter < numRows) {
		// Setting independent variables set and result set
		std::vector<double> tmp;
		for (int i = 0; i < inptCnt; ++i) {
			double value = stats[selectX[i] - 1];
			tmp.push_back(value);
			if (counter < numRowsInput)
				hx[_id(counter, i, inptCnt)] = value;
		}
		
		if (counter++ < numRowsInput) {
			dataX.push_back(tmp);
			dataY.push_back(stats[selectY - 1]);
			hy[counter] = stats[selectY - 1];
		}
		else {
			verifyX.push_back(tmp);
			verifyY.push_back(stats[selectY - 1]);
		}
	}
	printf("\n\n%d\n\n", counter);

	// Initialize Data Visualization
	//inputData(dataX, dataY);			// Input Data 
	//yStats = selectY;					// Hang over Y-category
	//getCategory(selectX, inptCnt);		// Hang over X-category
	//initGL(&argc, argv);				// Init OpenGL

	// Multiple Regression
	printf("\n[ Multiple Regression on Progress... ]\n");
	timer.onTimer(0);

	mr.fitIt(dataX, dataY, coeffs);
	timer.offTimer(0);
	printf("Completed function: f(x) = %f", coeffs[0]);
	for (int i = 0; i < inptCnt; ++i)
		printf(" + %f * x%d", coeffs[i + (int)1], i);
	printf("\n");

	// Multiple Regression Parallelized
	printf("\n[ Parallelized OpenMP Multiple Regression on Progress... ]\n");
	timer.onTimer(1);
	mrp.fitIt(dataX, dataY, coeffsP);
	timer.offTimer(1);
	printf("Completed function: f(x) = %f", coeffsP[0]);
	for (int i = 0; i < inptCnt; ++i)
		printf(" + %f * x%d", coeffsP[i + (int)1], i);
	printf("\n");

	// Multiple Regression CUDA
	printf("\n[ Parallelized  CUDA Multiple Regression on Progress... ]\n");
	timer.onTimer(2);
	cudaMemcpy(dx, hx, sizeof(double)*inptCnt*numRowsInput, cudaMemcpyHostToDevice);
	cudaMemcpy(dy, hy, sizeof(double)*numRowsInput, cudaMemcpyHostToDevice);

	kernelCall(dx, dy, dcoeffsP_2, inptCnt, matB);

	cudaMemcpy(coeffsP_2, dcoeffsP_2, sizeof(double)*inptCnt, cudaMemcpyDeviceToHost);

	timer.offTimer(2);
	printf("Completed function: f(x) = %f", coeffsP_2[0]);
	for (int i = 0; i < inptCnt; ++i)
		printf(" + %f * x%d", coeffsP_2[i + (int)1], i);
	printf("\n\n");

	// Verifying results with real dataset
	printf("\n\n[ Verifying with Dataset ]\n"); 
	printf("How many dataset do you want to verify? : ");
	while (~scanf("%d", &inpt)) {
		if (inpt > 0 && inpt < numRowsVerify) break;
		else if (inpt >= numRowsVerify) printf("I'm afraid there's only %d rows for verifying...\n", numRowsVerify);
		else printf("please select a proper number\n");
	} pressAny();

	printf("\nGiven Factors were \nX { ");
	for(int i = 0; i < inptCnt; ++i)
		printf("%s, ", statStr[selectX[i] - 1].c_str());
	printf("}, \nY { %s }\n\n", statStr[selectY - 1].c_str());

	bool isCorrect = true;

	for (int i = 0; i < inptCnt;i++) {
		if (ABS(coeffs[i] - coeffsP[i]) > EPSILON) {
			isCorrect = false;
		}
		if (ABS(coeffs[i] - coeffsP_2[i]) > EPSILON) {
			isCorrect = false;
		}
	}

	for (int i = 0; i < inpt; ++i) {
		//printf("X[%3d]{", i);
		//for (double d : verifyX[i])
		//	printf("%6.2f ", d);
		//printf("}");
		double serialResult = calMF(verifyX[i], coeffs);
		double parallelResult = calMF(verifyX[i], coeffsP);
		double parallelResult2 = calMF(verifyX[i], coeffsP_2);

		printf(" | Y[%3d](%8.2f) | f(x)=%12.3f (p)f(x)=%12.3f (p2)f(x)=%12.3f\n", i, verifyY[i], serialResult, parallelResult, parallelResult2);
	}
	printf("\n\n");
	if (isCorrect)
		printf("Result Correct\n");
	else 
		printf("Result Incorrect\n");
	timer.printTimer();
	double serialTime = timer.getTimer_ms(0);
	double parallelTime = timer.getTimer_ms(2);

	printf("\tx%.2f", serialTime / parallelTime);

	pressAny();

	cudaFreeHost(hx); cudaFreeHost(hy);
	cudaFree(dx); cudaFree(dy);

	for (int i = 0; i < numStats + 2; i++) {
		cudaFree(matB[i]);
	}

	return 0;
}
