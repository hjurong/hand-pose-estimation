#include "visualiser.h"
#include "fingermodel.h"
#include "thumbmodel.h"
#include "handmodel.h"
#include "observedmodel.h"
#include "PSO.h"
#include "costfunc.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace arma;
using namespace std;

mat spheres;
vec h_radii;
vec x0(26);

handmodel * hand_pointer;
observedmodel * observed_pointer;
costfunc *optfunc_pointer;
PSO *optimiser_pointer;

int num_p = 32;
int maxiter = 80;
int currentFrame = 0;

bool paused = true;

GLdouble viewer[] = {0.0, 0.0, 20.0};

string full_path;


/* Initialize OpenGL Graphics */
void initGL() {

    GLfloat light0_position[] = {1,1,1,0};
    GLfloat light0_ambient_color[] = {0.25,0.25,0.25,1};
    GLfloat light0_diffuse_color[] = {1,1,1,1};

    glPolygonMode(GL_FRONT,GL_FILL);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT,GL_AMBIENT_AND_DIFFUSE);

    glClearColor(0.0,0.0,0.0,0.0);
    glShadeModel(GL_SMOOTH);

    /* set up the light source */
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient_color);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse_color);

    /* turn lighting and depth buffering on */
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
}


/* Multiply two quaternions */
/* Use Eq. A-79 on p. 806 of Hearn and Baker */
void quaternion_multiply(float *out_q, float *q1, float *q2)
{

    float q[4];
    int i;

    // s = s1*s2 - v1.v2

    q[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3];

    // v = s1 v2 + s2 v2 + v1 x v2

    CROSS_PRODUCT(&q[1],&q1[1],&q2[1]);

    for(i=1;i<4;++i)
        q[i] += q1[0]*q2[i]+q2[0]*q1[i];

    // copy result to output vector

    for(i=0;i<4;++i)
        out_q[i] = q[i];
}

/* Use Eq. 5-107 on p. 273 of Hearn and Baker */
/* be aware that OpenGL uses transpose of the matrix */
void quaternion_to_matrix(float q[4], float M[4][4])
{
    float a,b,c,s;

    s = q[0];
    a = q[1];
    b = q[2];
    c = q[3];

    M[0][0] = 1 - 2*b*b - 2*c*c;
    M[1][0] = 2*a*b - 2*s*c;
    M[2][0] = 2*a*c + 2*s*b;

    M[0][1] = 2*a*b + 2*s*c;
    M[1][1] = 1 - 2*a*a - 2*c*c;
    M[2][1] = 2*b*c - 2*s*a;

    M[0][2] = 2*a*c - 2*s*b;
    M[1][2] = 2*b*c + 2*s*a;
    M[2][2] = 1 - 2*a*a - 2*b*b;

    M[0][3] = M[1][3] = M[2][3] = 0.0;
    M[3][0] = M[3][1] = M[3][2] = 0.0;
    M[3][3] = 1.0;
}

/* due to accumulating round-off error, it may be necessary to normalize */
/* this will ensure that the quaternion is truly unit */
void quaternion_normalize(float q[4])
{
    float mag=0;
    int i;

    for(i=0;i<4;++i)
        mag+=q[i]*q[i];

    mag = sqrt(mag);

    if(mag > EPSILON)
        for(i=0;i<4;++i)
            q[i] /= mag;
}

/* Handler for window-repaint event. Called back when the window first appears and
	whenever the window needs to be re-painted. */
void display_mainwindow() {

    int numwires = 10;

	float M[4][4];

	/* clear the display */
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

    // 
	quaternion_to_matrix(viewing_quaternion,M);
	glMultMatrixf(&M[0][0]);

    // spheres & h_radii are global

	int num_spheres = spheres.n_rows; // spheres.shape = (48,3)

	GLfloat cx = (GLfloat)spheres(0, 0);
	GLfloat cy = (GLfloat)spheres(0, 1);
	GLfloat cz = (GLfloat)spheres(0, 2);

    gluLookAt(viewer[0],viewer[1],viewer[2],cx,cy,cz,0.0,1.0,0.0);

    for (int i=1; i<num_spheres; i++) {

        GLfloat x = (GLfloat)spheres(i, 0);
		GLfloat y = (GLfloat)spheres(i, 1);
		GLfloat z = (GLfloat)spheres(i, 2);

		glTranslatef(x, y, z);  // Move to new position

		// set colour according to finger
		if (i >= 38) {
			glColor3f(0.8471, 0.7490, 0.8471); // little finger
		}
		else if (i >= 28) {
			glColor3f(0.9725, 0.9725, 1.0000);
		}
		else if (i >= 18) {
			glColor3f(0.9333, 0.9098, 0.6667);
		}
		else if (i >= 8) {
			glColor3f(0.1333, 0.5451, 0.1333);
		}
		else {
			glColor3f(0.6863, 0.9333, 0.9333);
		}

        GLdouble sphR = (GLdouble)h_radii(i);

        glutSolidSphere(sphR,numwires,numwires);

        // move back to (0,0,0)
        glTranslatef(-x, -y, -z);  // Move right and into the screen

    }

    mat pointcloud = *(observed_pointer->get_ptncloud());


    int numptns = pointcloud.n_rows; // pointcloud.shape = (nptns, 3)
    glPointSize(5);
    glBegin(GL_POINTS);
    for (int n=0; n<numptns; n++) {
    	GLdouble x = (GLdouble)pointcloud(n,0);
    	GLdouble y = (GLdouble)pointcloud(n,1);
    	GLdouble z = (GLdouble)pointcloud(n,2);
    	glColor3f(0.8157, 0.1255, 0.5647);
    	glVertex3d(x,y,z);
    }
    glEnd();


	glFlush();


	glutSwapBuffers();  // Swap the front and back frame buffers (double buffering)


}


void mainMenu (int id)
/* This is the callback function for the main menu. */
{
    switch (id)
    {
    case 1 : /* reset default values */
    	viewer[0] = 0.0;
    	viewer[1] = 0.0;
    	viewer[2] = 20.0;

    	viewing_quaternion[0] = 1.0;
    	viewing_quaternion[1] = 0.0;
    	viewing_quaternion[2] = 0.0;
    	viewing_quaternion[3] = 0.0;
        break;
    case 2 : /* clear the screen */
        paused = !paused;
        break;
    case 3 : /* exit the program */
        exit (0);
        break;
    default : /* in case none of the above occur */
        break;
    }

}

void subMenu1 (int id)
/* This is the callback function for the color menu. */
{

}

void subMenu2 (int id)
/* This is the callback function for the size menu. */
{

}


void update_frame() {

    if (currentFrame < 400 && !paused) {
        stringstream ss;
        ss << setw(6) << setfill('0') << currentFrame++;
        string next_frame_num = ss.str();

        string nextframe_name = next_frame_num + "_depth.bin";
        string nextframe_jpeg = next_frame_num + "_depth.jpg";

        cv::Mat image= cv::imread(full_path+nextframe_jpeg, CV_LOAD_IMAGE_COLOR);
        cv::imshow("Display window", image );

        observed_pointer->next_frame(nextframe_name);

        optimiser_pointer->refine_init_pose(x0, *optfunc_pointer);

        vec bestp = zeros<vec>(26);
    //      optimiser.pso_optimise(optfunc, x0, num_p, bestp);
//        optimiser_pointer->pso_optimise(*optfunc_pointer, x0, num_p, bestp);
        optimiser_pointer->pso_evolve(*optfunc_pointer, x0, num_p, bestp);

        double c = optfunc_pointer->cal_cost(bestp);
        string label = "frame" + next_frame_num + "-cost: ";
        cout << label << c << endl;

        x0 = bestp;

        hand_pointer->build_hand_model(x0, spheres);

    	glutPostRedisplay(); // Post re-paint request to activate display()

    	// saving visualisation as a jpg file
//    	cv::Mat img(480, 640, CV_8UC3);
//		glPixelStorei(GL_PACK_ALIGNMENT, (img.step & 3) ? 1 : 4);
//		glReadPixels(0, 0, img.cols, img.rows, GL_RGB, GL_UNSIGNED_BYTE, img.data);
//
//		string save_fname = "pose" + next_frame_num + ".jpg";
//		cv::Mat flipped(480, 640, CV_8UC3);
//		cv::flip(img, flipped, 0);
//		cv::imwrite(save_fname, flipped);

    }


}

/* This is the keyboard callback function. Keys change the viewer's position as well as turn
   rotation on and off. */
void keys(unsigned char key, int x, int y)
{
   if(key == 'x') viewer[0]-= 1.0;
   if(key == 'X') viewer[0]+= 1.0;
   if(key == 'y') viewer[1]-= 1.0;
   if(key == 'Y') viewer[1]+= 1.0;
   if(key == 'z') viewer[2]-= 1.0;
   if(key == 'Z') viewer[2]+= 1.0;
   if(key == 'p') paused = !paused;
   glutPostRedisplay(); // Post re-paint request to activate display()

}


void mouseButton(int button, int state, int x, int y)
{

    currentButton = button;
    if (button == GLUT_LEFT_BUTTON){
        switch (state) {
            case GLUT_DOWN:
                last_x = x;
                last_y = y;
                break;
        }
    }
}

/* mouse motion callback */
void mouseMotion(int x, int y)
{
    float dx,dy;
    float rotation_axis[3], mag, q[4];
    float viewing_delta = PI/360.0;  /* 0.5 degrees */
    float s,c;

    switch(currentButton) {

        case GLUT_LEFT_BUTTON:

        	/* vector in direction of mouse motion */
        	dx = x - last_x;
        	dy = y - last_y;

        	/* spin around axis by small delta */
        	mag = sqrt(dx*dx+dy*dy);
        	rotation_axis[0]=dy/mag;
        	rotation_axis[1]=dx/mag;
        	rotation_axis[2]=0.0;

        	/* calculate the appropriate quaternion */
        	s = sin(0.5*viewing_delta);
        	c = cos(0.5*viewing_delta);

        	q[0] = c;
        	q[1] = s*rotation_axis[0];
        	q[2] = s*rotation_axis[1];
        	q[3] = s*rotation_axis[2];

        	quaternion_multiply(viewing_quaternion,q,viewing_quaternion);

        	/* normalize, to counteract accumulating round-off error */
        	quaternion_normalize(viewing_quaternion);

        	/* save current x,y as last x,y */
        	last_x = x;
        	last_y = y;
            break;
        }
    /* redisplay */
    glutPostRedisplay();
}


/* Handler for window re-size event. Called back when the window first appears and
	whenever the window is re-sized with its new width and height */
void reshape_mainwindow(GLsizei width, GLsizei height) {  

    double ratio;

    /* Prevent a divide by zero, when window is too short
    	(you cant make a window of zero width). */
    if(height == 0) height = 1;

    ratio = 1.0f * width / height;

    /* Reset the coordinate system before modifying */
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    /* Set the viewport to be the entire window */
    glViewport(0, 0, width, height);

    /* Set the clipping volume */
    gluPerspective(25,ratio,0.1,100);

    /* Camera positioned at (0,0,6), look at point (0,0,0), Up Vector (0,1,0) */
    GLfloat x = (GLfloat)spheres(0, 0);
	GLfloat y = (GLfloat)spheres(0, 1);
	GLfloat z = (GLfloat)spheres(0, 2);

    gluLookAt(viewer[0],viewer[1],viewer[2],x,y,z,0.0,1.0,0.0);

    glMatrixMode(GL_MODELVIEW);
}


void gl_visualise() {

    // dummy arguments for glutInit()
    int argc = 1;
    char *argv[1] = {(char*)"init"};

	vec hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);
	tbnum << 2 << 2 << 2 << 2 << endr;
	fgnum << 4 << 2 << 2 << 2 << endr;
	spc << -1.86 << -1.86 << 0 << 1.91 << 3.84 << endr; // digit spacing
	hcmc << 150 << 107.5 << 89.8 << 76.5 << 59.6 << endr; // cmc angles
	x0 << 0 << -10 << -40 << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
	   << 9 << 6 << 1  << 9  << 8 << 7 << 4 << 8 << 7 << 6 << 2
	   << 7 << 7 << 7  << endr; // initial pose

	// load hand geometry
	string handparam_path = "misc";
	string file1 = handparam_path + "/hgeo.dat"; // unit = mm
	string file2 = handparam_path + "/rad.dat";
	hgeo.load(file1, raw_ascii);
	hrad.load(file2, raw_ascii);

	hrad = hrad / 10.; // convert to cm
	hgeo = hgeo / 10.; // convert to cm

	h_radii = hrad;
	spheres = zeros<mat>(48,3);

	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad); // initialise hand

	hand_pointer = &hand;

	string home = "../";
	string subj = "handModelling/Release_2014_5_28/Subject1/";
	string objt = "000000_depth.bin";

	full_path = home + subj;

	double focal = 241.42;
	int imgH = 320, imgW = 240;
	bool conv_to_cm = true;
	bool downsample = true;

	observedmodel observation;
	observation.init_observation(full_path, objt, conv_to_cm, imgW, imgH,
								 focal, downsample);
	observed_pointer = &observation;

	costfunc optfunc(&hand, &observation);

	optfunc_pointer = &optfunc;

	vec temp(4);
	temp << 15 << 90 << 110 << 90 << endr;
	vec ub = zeros<vec>(26);
	ub.rows(0,2).fill(180);
	ub.rows(3,5).fill(100);
	ub.rows(6,9) = temp;
	ub.rows(10, 13) = temp;
	ub.rows(14, 17) = temp;
	ub.rows(18, 21) = temp;
	ub.rows(22, 25) = temp;

	temp << -15 << 0 << 0 << 0 << endr;
	vec lb = zeros<vec>(26);
	lb.rows(0,2).fill(-180);
	lb.rows(3,5).fill(-100);
	lb.rows(6,9) = temp;
	lb.rows(10, 13) = temp;
	lb.rows(14, 17) = temp;
	lb.rows(18, 21) = temp;
	lb.rows(22, 25) = temp;

	vec std = zeros<vec>(26);
	std.rows(0, 2).fill(9.0);
	std.rows(3, 5).fill(7.0);
	std.rows(6,25).fill(9.0);

	double w = 0.7298;
	double c1 = 1.49618;
	double c2 = 1.49618;
	double minstep = 1e-8;
	double minfunc = 1e-8;

	PSO optimiser;
	optimiser.set_pso_params(ub, lb, std, w, c1, c2,
							 maxiter, minstep, minfunc);
	optimiser_pointer = &optimiser;

	stringstream ss;
	ss << setw(6) << setfill('0') << currentFrame++;
	string next_frame_num = ss.str();

	string nextframe_name = next_frame_num + "_depth.bin";
	string nextframe_jpeg = next_frame_num + "_depth.jpg";

	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
	cv::moveWindow("Display window", 10, 50);

	cv::Mat image= cv::imread(full_path+nextframe_jpeg, CV_LOAD_IMAGE_COLOR);
	cv::imshow( "Display window", image );

	observed_pointer->next_frame(nextframe_name);

	optimiser_pointer->refine_init_pose(x0, optfunc);

	vec bestp = zeros<vec>(26);

//	optimiser_pointer->pso_optimise(optfunc, x0, num_p, bestp);
	optimiser_pointer->pso_evolve(optfunc, x0, num_p, bestp);

	double c = optfunc.cal_cost(bestp);
	string label = "frame" + next_frame_num + "-cost: ";
	cout << label << c << endl;

	x0 = bestp;

	hand_pointer->build_hand_model(x0, spheres);


    glutInit(&argc, argv); // Initialize GLUT
    glutInitDisplayMode(GLUT_DOUBLE| GLUT_RGBA | GLUT_DEPTH); // Enable double buffered mode
    glutInitWindowSize(640, 480);   // Set the window's initial width & height
    glutCreateWindow("Real-time Visual"); // Create window with the given title
    glutDisplayFunc(display_mainwindow);  // Register callback handler for window re-paint event
    glutReshapeFunc(reshape_mainwindow);  // Register callback handler for window re-size event
    glutMotionFunc(mouseMotion);
    glutMouseFunc(mouseButton);
    glutKeyboardFunc(keys);
    initGL();                       // Our own OpenGL initialization


    int sub_menu1, sub_menu2;
    sub_menu1 = glutCreateMenu (subMenu1);
    glutAddMenuEntry("Info1", 1);
    glutAddMenuEntry("Info2", 2);
    glutAddMenuEntry("Info3", 3);
    glutAddMenuEntry("Info4", 4);

    sub_menu2 = glutCreateMenu (subMenu2);
    glutAddMenuEntry("1", 1);
    glutAddMenuEntry("2", 2);
    glutAddMenuEntry("3", 3);
    
    glutCreateMenu(mainMenu);
    glutAddSubMenu("Help", sub_menu1);
    glutAddSubMenu("More", sub_menu2);
    glutAddMenuEntry("Reset defaults", 1);
    glutAddMenuEntry("Pause", 2);
    glutAddMenuEntry("Exit", 3);
    glutAttachMenu (GLUT_RIGHT_BUTTON);

//    glutTimerFunc(0, timer, 0);     // First timer call immediately [NEW]
    glutIdleFunc(update_frame);
    glutMainLoop();                 // Enter the infinite event-processing loop
}
