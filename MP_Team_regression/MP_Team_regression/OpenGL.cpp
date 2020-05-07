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
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	glutInitDisplayMode(GLUT_RGBA);
	glutInitWindowSize(WIDTH, HEIGHT);			// Window Size
	glutCreateWindow("Mid Term Project");		// Window Name
	glutDisplayFunc(dataVisualization);			// Data Visualization

	// Menu
	menu();

	glutMainLoop();
}

/* Render Scene Settings */
void dataVisualization() {
	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	// 1) Draw Coordinate System
	drawCoordinateSystem();

	// 2) Draw all points of data
	drawTotalPoints();

	// 3) Title
	std::string title = "All Data";
	glColor3f(0.0f, 0.0f, 0.0f);
	glRasterPos3f(0.60f, 0.95f, 0.0f);
	for (auto c : title) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
	}

	// 4) y-Label
	float tmpColor[3] = { 0.0f, 0.0f, 0.0f };
	drawLabels(-0.95f, 0.95f, yStats - 1, tmpColor, 0);

	glutSwapBuffers();
	glFlush();
}

/* Visualization of Each X-data */
void eachDataVisualization() {
	// Set background color
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glLoadIdentity();

	// 1) Draw Coordinate System
	drawCoordinateSystem();

	// 2) Draw all points of data
	drawEachPoints();

	// 3) y-Label
	float tmpColor[3] = { 0.0f, 0.0f, 0.0f };
	drawLabels(-0.95f, 0.95f, yStats - 1, tmpColor, 0);

	glutSwapBuffers();
	glFlush();
}

/* Draw coordinate system & points */
void drawCoordinateSystem() {
	glColor3f(0.0f, 0.0f, 0.0f);

	// the range of Window is between -1.0 and 1.0

	// X-coord
	glBegin(GL_LINE_LOOP);
	glVertex3f(0.95, -0.95, 0.0);
	glVertex3f(-0.95, -0.95, 0.0);
	glEnd();

	// Y-coord
	glBegin(GL_LINE_LOOP);
	glVertex3f(-0.95, 0.95, 0.0);
	glVertex3f(-0.95, -0.95, 0.0);
	glEnd();

}

void drawTotalPoints() {
	// Randomize
	srand((unsigned)time(NULL));

	// Size of Point
	glPointSize(4.0f);

	int curCategory = -1;						// category index
	float y = 0.90f;							// label y-coord
	std::vector<std::vector<float>> colors;		// colors

	// Setting Colors
	for (int i = 0; i < category.size(); i++) {
		std::vector<float> tmp;

		// For randomizing the color by category to distinguish
		for (int i = 0; i < 3; i++) {
			tmp.push_back((rand() % 100) / 100.0);
		}
		colors.push_back(tmp);

		float color[3] = { colors[i][0], colors[i][1], colors[i][2] };

		// Print Labels
		drawLabels(0.6f, y, i, color, 1);
		y -= 0.05f;
	}

	// Draw points of data
	glBegin(GL_POINTS);
	for (auto data : dataInfo) {
		if (curCategory < data.getCategory())
			curCategory++;

		glColor3f(
			colors[curCategory][0],
			colors[curCategory][1],
			colors[curCategory][2]
		);
		glVertex3f(data.getX(), data.getY(), 0.0);
	}
	glEnd();
}

void drawEachPoints() {
	// Randomize
	srand((unsigned)time(NULL));

	// Size of point
	glPointSize(4.0f);

	// Label
	float color[3] = { 0.0, };
	for (int i = 0; i < 3; i++) color[i] = (rand() % 100) / 100.0;
	drawLabels(0.60f, 0.95f, currMenuOper, color, 1);

	// Draw points of data
	glBegin(GL_POINTS);
	for (auto data : eachDataInfo[currMenuOper]) {
		glColor3f(color[0], color[1], color[2]);
		glVertex3f(data.getX(), data.getY(), 0.0);
	}
	glEnd();
}

void drawLabels(float x, float y, int index, float* color, int mod) {	// category index & color information
	std::string xLabel[39] = {
		"blueWins",
		"blueWardsPlaced", "blueWardsDestroyed", "blueFirstBlood", "blueKills", "blueDeaths", "blueAssists",
		"blueEliteMonsters", "blueDragons", "blueHeralds", "blueTowersDestroyed", "blueTotalGold", "blueAvgLevel",
		"blueTotalExperience", "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled", "blueGoldDiff", "blueExperienceDiff",
		"blueCSPerMin", "blueGoldPerMin",
		"redWardsPlaced", "redWardsDestroyed", "redFirstBlood", "redKills", "redDeaths", "redAssists",
		"redEliteMonsters", "redDragons", "redHeralds", "redTowersDestroyed", "redTotalGold", "redAvgLevel",
		"redTotalExperience", "redTotalMinionsKilled", "redTotalJungleMinionsKilled", "redGoldDiff", "redExperienceDiff",
		"redCSPerMin", "redGoldPerMin"
	};

	std::string label;
	if (mod)
		label = xLabel[category[index]];	// mod 1 is X-label
	else
		label = xLabel[index];				// mod 0 is Y-label

	glColor3f(color[0], color[1], color[2]);
	glRasterPos3f(x, y, -0.1f);
	for (auto c : label) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
	}
	//glutBitmapString(GLUT_BITMAP_HELVETICA_12, (const unsigned char*)label.c_str());
}

/* Data Setting */
void inputData(std::vector<std::vector<double>> x, std::vector<double> y) {
	std::vector<std::vector<double>> nx = vectorNormalization(x);
	std::vector<double>				 ny = vectorNormalization(y);

	/* Total Data */
	// if col size of X  is same as row of Y size
	for (int i = 0; i < nx.size(); i++) {
		for (int j = 0; j < nx[i].size(); j++) {
			// Point(category, index, pair<coordinate>)
			Point tmp(j, i, std::pair<float, float>(nx[i][j], ny[i]));
			dataInfo.push_back(tmp);
		}
	}

	sort(dataInfo.begin(), dataInfo.end(), cmp);

	/* Each Data */
	// Transpose X
	std::vector<std::vector<double>> transposeX(x[0].size(), std::vector<double>());
	for (int i = 0; i < x.size(); i++) {
		for (int j = 0; j < x[i].size(); j++) {
			transposeX[j].push_back(x[i][j]);
		}
	}

	for (int i = 0; i < transposeX.size(); i++) {
		std::vector<double> px = vectorNormalization(transposeX[i]);
		std::vector<Point> tmpPoint;
		for (int j = 0; j < px.size(); j++) {
			Point tmp(i, j, std::pair<float, float>(px[j], ny[j]));
			tmpPoint.push_back(tmp);
		}
		eachDataInfo.push_back(tmpPoint);
	}

}

bool cmp(Point a, Point b) {
	if (a.getCategory() >= b.getCategory()) return false;
	else return true;
}

// 1D-vector version - for Y vector or each X vector
// Make value of vector between 0.0 and 1.0
std::vector<double> vectorNormalization(std::vector<double> v) {
	double maxV = 0;
	double minV = DBL_MAX;
	double avg = 0.0;

	// Find max & min value
	for (auto val : v) {
		if (maxV < val) maxV = val;
		if (minV > val) minV = val;
		avg += val;
	}

	avg /= v.size();

	// Normalize all elements of vector
	std::vector<double> tmp;
	double t;

	for (auto val : v) {
		double t = (val - minV) / (double)(maxV - minV);
		tmp.push_back(2.0f * t - 0.92f);
	}

	return tmp;
}

// 2D-vector version - for all X vector
std::vector<std::vector<double>> vectorNormalization(std::vector<std::vector<double>> v) {
	double maxV = 0;
	double minV = DBL_MAX;

	// Find max & min value
	for (auto rows : v) {
		for (auto val : rows) {
			if (maxV < val) maxV = val;
			if (minV > val) minV = val;
		}
	}

	// Normalize all elements of vector
	std::vector<std::vector<double>> normal;
	double t;

	for (int i = 0; i < v.size(); i++) {
		std::vector<double> tmp;
		for (int j = 0; j < v[i].size(); j++) {
			t = (v[i][j] - minV) / (double)(maxV - minV);
			// resize to fit window size 
			tmp.push_back(2.0f * t - 0.92f);
		}
		normal.push_back(tmp);
	}

	return normal;
}

// Take category information to OpenGL
void getCategory(int* c, int size) {
	for (int i = 0; i < size; i++) {
		category.push_back(c[i] - 1);
	}
}

/* Menu */
void menu() {
	std::string xLabel[39] = {
		"blueWins",
		"blueWardsPlaced", "blueWardsDestroyed", "blueFirstBlood", "blueKills", "blueDeaths", "blueAssists",
		"blueEliteMonsters", "blueDragons", "blueHeralds", "blueTowersDestroyed", "blueTotalGold", "blueAvgLevel",
		"blueTotalExperience", "blueTotalMinionsKilled", "blueTotalJungleMinionsKilled", "blueGoldDiff", "blueExperienceDiff",
		"blueCSPerMin", "blueGoldPerMin",
		"redWardsPlaced", "redWardsDestroyed", "redFirstBlood", "redKills", "redDeaths", "redAssists",
		"redEliteMonsters", "redDragons", "redHeralds", "redTowersDestroyed", "redTotalGold", "redAvgLevel",
		"redTotalExperience", "redTotalMinionsKilled", "redTotalJungleMinionsKilled", "redGoldDiff", "redExperienceDiff",
		"redCSPerMin", "redGoldPerMin"
	};

	// Add menu list including the name of x-data
	int i;
	glutCreateMenu(menuOperator);
	for (i = 0; i < category.size(); i++) {
		glutAddMenuEntry(xLabel[category[i]].c_str(), i);
	}

	glutAddMenuEntry("All", i);
	glutAddMenuEntry("Exit", i + 1);
	glutAttachMenu(GLUT_RIGHT_BUTTON);	// Show menu when right click 
}

void menuOperator(int oper) {
	if (oper == category.size()) {				// Show all data
		glutDisplayFunc(dataVisualization);
	}
	else if (oper == category.size() + 1) {		// Exit data visualization
		glutLeaveMainLoop();
	}
	else {										// Show each data
		currMenuOper = oper;
		glutDisplayFunc(eachDataVisualization);
	}

	glutPostRedisplay();
}

void printString(std::string txt, float x, float y, float z) {
	glRasterPos3f(x, y, z);
	for (auto c : txt) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
	}
}

/* Print for Deubg */
void printVector(int i) {
	std::cout << "x : " << dataInfo.at(i).getX() << " y : " << dataInfo.at(i).getY() << std::endl;
}