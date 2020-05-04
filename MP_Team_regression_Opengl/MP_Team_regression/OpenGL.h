#pragma once
#include <GL/glew.h>
#include <GL/glut.h>
#include <Windows.h>
#include <vector>
#include <iostream>			// For debug

/* Opengl Windows Setting Variables */
// Windows
#define WIDTH	1280
#define HEIGHT	720

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

	// Operator

};

/* Opengl Functions */
void initGL(int* argc, char** argv);
void renderScene();
void drawCoordinateSystem();
void drawPoints();

/* Data Contorl */
void inputData(std::vector<double> x, std::vector<double> y);
std::vector<double> vectorNormalization(std::vector<double> v);

/* Global Variable */
extern std::vector<Point> dataInfo;
extern std::vector<double> coeffs;