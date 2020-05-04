#include "OpenGL.h"

/* Point Class Method */
Point::Point()
{
	category = -1;
	index = -1;
	coordinate = std::pair<float, float>(0.0f, 0.0f);
}

Point::Point(int _category,
	int _index,
	std::pair<float, float> _coordinate)
{
	category = _category;
	index = _index;
	coordinate = _coordinate;
}

Point::~Point() {}

/* Point get function */
int Point::getCategory() { return category; }
int Point::getIndex() { return index; }
float Point::getX() { return coordinate.first; }
float Point::getY() { return coordinate.second; }
std::pair<float, float> Point::getCoordinate() { return coordinate; }

/* Point set function */
void Point::setCategory(int c) { category = c; }
void Point::setIndex(int i) { index = i; }
void Point::setX(float x) { coordinate.first = x; }
void Point::setY(float y) { coordinate.second = y; }
void Point::setCoordinate(std::pair<float, float> c) { coordinate = c; }

/* Init OpenGL */
void initGL(int* argc, char** argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(WIDTH, HEIGHT);			// Window Size
	glutCreateWindow("Mid Term Project");		// Window Name
	glutDisplayFunc(renderScene);				// Start Window Display

	// Mouse

	// Menu

	// Info

	glutMainLoop();
}

/* Render Scene Settings */
void renderScene() {
	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	// 1) Draw Coordinate System
	drawCoordinateSystem();

	// 2) Draw all points of data
	drawPoints();

	// 3) When the calculation starts, turn on & off vertices at each cal. moment


	// 4) If all task finish, 
	//	  [1] change a new board to draw result 
	//	  [2] just draw on the board with the vertices

	glutSwapBuffers();

	glFlush();
}

/* Draw coordinate system & points */
void drawCoordinateSystem() {
	glColor3f(0.0f, 0.0f, 0.0f);

	// x√‡
	glBegin(GL_LINE_LOOP);
	glVertex3f(1.0, -0.95, 0.0);			// input-1~1 range of value
	glVertex3f(-1.0, -0.95, 0.0);
	glEnd();

	// Y√‡
	glBegin(GL_LINE_LOOP);
	glVertex3f(-0.95, 1.0, 0.0);
	glVertex3f(-0.95, -1.0, 0.0);
	glEnd();
	
}

void drawPoints() {
	glColor3f(0.2f, 0.2f, 0.2f);
	glPointSize(5.0f);

	// Debug
	std::cout << "Drawing Points" << std::endl;

	// Consider frame range is -1 ~ 1
	glBegin(GL_POINTS);
	for (auto data : dataInfo) {
		glVertex3f(data.getX(), data.getY(), 0.0);

		std::cout << data.getX() << " " << data.getY() << std::endl;
	}
	glEnd();
}

/* Data Setting */
void inputData(std::vector<double> x, std::vector<double> y) {
	std::vector<double> nx = vectorNormalization(x);
	std::vector<double> ny = vectorNormalization(y);

	// if data x, y size is same
	for (int i = 0; i < x.size(); i++) {
		// Point(category, index, pair<coordinate>)
		Point tmp(0, i, std::pair<float, float>(nx[i], ny[i]));
		dataInfo.push_back(tmp);
	}
}

// Make value of vector between 0.0 and 1.0
std::vector<double> vectorNormalization(std::vector<double> v) {
	double maxV = 0;
	double minV = DBL_MAX;

	// Find max & min value
	for (auto val : v)
	{
		if (maxV < val) maxV = val;
		if (minV > val) minV = val;
	}

	std::vector<double> tmp;

	for (auto val : v) {
		double t = (val - minV) / (maxV - minV);
		tmp.push_back(2 * t - 0.95);
	}

	return tmp;
}


/* Print for Deubg */
void printVector(int i) {
	std::cout << "x : " << dataInfo.at(i).getX() << " y : " << dataInfo.at(i).getY() << std::endl;
}


/* After Calculation */
void inputCoeffs(std::vector<double> c) {

}