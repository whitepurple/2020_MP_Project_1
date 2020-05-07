#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <Windows.h>
#include <vector>
#include <time.h>
#include <string>
#include <algorithm>
#include <iostream>			// For debug

/* Opengl Windows Setting Variables */
// Windows
#define WIDTH	720
#define HEIGHT	480

typedef const unsigned char CUCHAR;

/* CLASS */
class Point {
private:
	int category;							// data category
	int index;								// In category, data index
	std::pair<float, float> coordinate;		// Real value of data in CSV file

public:
	Point();
	Point(int _category, 
		  int _index,
		  std::pair<float, float> _coordinate);
	~Point();

	// Getter
	int getIndex();
	int getCategory();
	float getX();
	float getY();
	std::pair<float, float> getCoordinate();

	// Setter
	void setIndex(int i);
	void setCategory(int c);
	void setX(float x);
	void setY(float y);
	void setCoordinate(std::pair<float, float> c);
};

/* Opengl Functions */
void initGL(int* argc, char** argv);
void dataVisualization();				// draw all data visualization
void eachDataVisualization();			// draw each data visualization
void drawCoordinateSystem();			// draw coordinate system
void drawTotalPoints();
void drawEachPoints();
void drawLabels(float x, float y, int index, float* color, int mod);

/* Data Contorl */
void inputData(std::vector<std::vector<double>> x, std::vector<double> y);
bool cmp(Point a, Point b);			// for Sorting
std::vector<double> vectorNormalization(std::vector<double> v);
std::vector<std::vector<double>> vectorNormalization(std::vector<std::vector<double>> v);
void getCategory(int* c, int size);

/* Global Variable */
extern std::vector<Point> dataInfo;
extern std::vector<std::vector<Point>> eachDataInfo;
extern std::vector<int> category;
extern int yStats;
extern int currMenuOper;

/* Menu */
void menu();
void menuOperator(int oper);
