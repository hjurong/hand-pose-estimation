#ifndef VISUALISER_H
#define VISUALISER_H

#include "handmodel.h"
#include <armadillo>
#include <iostream>
#include <string>
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace arma;
using namespace std;


static int last_x, last_y;
static int currentButton;
static float viewing_quaternion[4] = {1,0,0,0};

#define PI 3.1415927

/* floating tolerance */
#define EPSILON 0.0001

/* the amount the angle changes each arrow press */
#define DELTA_ANGLE 1.0

/* rotation axis */
#define AXIS_X 0
#define AXIS_Y 1
#define AXIS_Z 2

/* default window dimensions */
#define DEFAULT_WINDOW_WIDTH 512
#define DEFAULT_WINDOW_HEIGHT 512

#define CROSS_PRODUCT(O,A,B) {(O)[0] = (A)[1]*(B)[2]-(A)[2]*(B)[1]; \
                              (O)[1] = (A)[2]*(B)[0]-(A)[0]*(B)[2]; \
                              (O)[2] = (A)[0]*(B)[1]-(A)[1]*(B)[0];}






/* gl callbacks */
void initGL();
void update_frame();
void reshape_mainwindow(GLsizei width, GLsizei height);
void display_mainwindow();
void keys(unsigned char key, int x, int y);
void special(int k, int x, int y);
void mouseMotion(int x, int y);
void mouseButton(int button, int state, int x, int y);
void mainMenu(int id);
void subMenu1(int id);
void subMenu2(int id);


/* quaternion.c */
void quaternion_multiply(float *out_q, float *q1, float *q2);
void quaternion_to_matrix(float q[4],float M[4][4]);
void quaternion_normalize(float q[4]);



void gl_visualise();




	


	


#endif
