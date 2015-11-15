/* test program for fingermodel.cpp */

#include "fingermodel.h"
#include "thumbmodel.h"
#include "handmodel.h"
#include "observedmodel.h"
#include "PSO.h"
#include "costfunc.h"
#include "visualiser.h"

#include <sys/stat.h>
#include <time.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GL/glut.h>  // GLUT, includes glu.h and gl.h

using namespace arma;
using namespace std;

void test_full() {
	/*
	 * run a test on multiple test frames
	 */

	// initial parameters
	vec x0(26), hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);
	tbnum << 2 << 2 << 2 << 2 << endr;
	fgnum << 4 << 2 << 2 << 2 << endr;
	spc   << -1.86 << -1.86 << 0    << 1.91 << 3.84 << endr; // digit spacing
	hcmc  << 150   << 107.5 << 89.8 << 76.5 << 59.6 << endr; // cmc angles
	x0 << 0 << -10 << -40   << 0 	<< 3 	<< 32 	<< 6 << 9 << 8 << 9 << 3 << 9
	   << 9 << 6   << 1     << 9  	<< 8 	<< 7 	<< 4 << 8 << 7 << 6 << 2
	   << 7 << 7   << 7  	<< endr; // initial pose

	// load hand geometry; predefined from data set
	string handparam_path = "misc";
	string file1 		  = handparam_path + "/hgeo.dat"; // unit = mm
	string file2 		  = handparam_path + "/rad.dat";
	hgeo.load(file1, raw_ascii);
	hrad.load(file2, raw_ascii);

	hrad = hrad / 10.; // convert to cm
	hgeo = hgeo / 10.; // convert to cm

	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad); // initialise hand

	// specify where the data set is
	string home = "../";
	string subj = "handModelling/Release_2014_5_28/Subject1/";
	string objt = "000000_depth.bin";
	string full = home + subj;

	// initialise observed model
	double focal = 241.42;
	int imgH = 320, imgW = 240;
	bool conv_to_cm = true;
	bool downsample = true;

	observedmodel observation;
	observation.init_observation(full, objt, conv_to_cm, imgW, imgH,
								 focal, downsample);

	// initialise cost function
	costfunc optfunc(&hand, &observation);

	// set upper and lower bounds for every dimension of theta
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

	// initialise optimiser
	double w = 0.7298;
	double c1 = 1.49618;
	double c2 = 1.49618;
	int maxiter = 200;
	int num_p = 32;
	double minstep = 1e-8;
	double minfunc = 1e-8;

	PSO optimiser;
	optimiser.set_pso_params(ub, lb, std, w, c1, c2,
							 maxiter, minstep, minfunc);

	// set number of frames to optimise
	int numframes = 10;
	vec fitness = zeros<vec>(numframes); // container for fitness

	for (int frame=0; frame < numframes; frame++) {
		stringstream ss;
		ss << setw(6) << setfill('0') << frame;
		string next_frame_num = ss.str();

		string nextframe_name = next_frame_num + "_depth.bin";

		observation.next_frame(nextframe_name);

		optimiser.refine_init_pose(x0, optfunc);

		vec bestp = zeros<vec>(26);

		optimiser.pso_evolve(optfunc, x0, num_p, bestp);

		double c = optfunc.cal_cost(bestp);
		string label = "frame" + next_frame_num + "-cost: ";
		cout << label << c << endl;

		fitness(frame) = c; // store fitness for current frame

		x0 = bestp;

	}

	// uncomment to output fitness data
//	fitness.save("fitness_error.dat", raw_ascii);


}

void test_PSO(costfunc &optfunc) {
	/*
	 * test the optimiser
	 *
	 * can change the initial pose vector and upper/lower bound parameters
	 * to test the effect
	 */
	vec x0(26);
	x0 << 0 <<-10<< -40<< 0 << 3 << 32<< 6 << 9 << 8 << 9 << 3 << 9
	   << 9 << 6 << 1  << 9 << 8 << 7 << 4 << 8 << 7 << 6 << 2
	   << 7 << 7 << 7  << endr;

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
	std.rows(0, 2).fill(5.0);
	std.rows(3, 5).fill(3.0);
	std.rows(6,25).fill(5.0);

	double w = 0.7298;
	double c1 = 1.49618;
	double c2 = 1.49618;
	int maxiter = 200;
	int num_p = 32;
	double minstep = 1e-8;
	double minfunc = 1e-8;

	PSO optimiser;
	optimiser.set_pso_params(ub, lb, std, w, c1, c2,
							 maxiter, minstep, minfunc);

	optimiser.refine_init_pose(x0, optfunc);

	vec bestp = zeros<vec>(26);

	optimiser.pso_evolve(optfunc, x0, num_p, bestp);

	cout << x0 << endl;
	double b = optfunc.cal_cost(x0);
	cout << b << endl;
	cout << bestp << endl;
	double c = optfunc.cal_cost(bestp);
	cout << c << endl;

}

void test_cost(handmodel *hand, observedmodel *observation) {
	vec hthe(26);
	hthe << 0 <<-10 << -40 << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
		 << 9 << 6  << 1   << 9 << 8 << 7  << 4 << 8 << 7 << 6 << 2
		 << 7 << 7  << 7   << endr;

	costfunc costfunction(hand, observation);

	mat::fixed<48,3> spheresM;
	hand->build_hand_model(hthe, spheresM);

}


void test_observedmodel(observedmodel &observation, bool downsample) {

	string home = "../";
	string path = "handModelling/Release_2014_5_28/Subject1/";
	string objt = "000004_depth.bin";
	string full = home + path;

	double focal = 241.42;
	int imgH = 320, imgW = 240;
	bool conv_to_cm = true;
	mat ptncloud;

	observation.init_observation(full, objt, conv_to_cm, imgW, imgH,
								 focal, downsample); // already calls depth_to_ptncloud
}


void test_visualiser() {

	gl_visualise();

}


handmodel test_handmodel() {
	///////////////////////////////////////////////////////////
	//// handmodel testing
	///////////////////////////////////////////////////////////
	vec hthe(26), hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);

	// set default parameters
	tbnum << 2 << 2 << 2 << 2 << endr;
	fgnum << 4 << 2 << 2 << 2 << endr;
	spc   << -1.86 << -1.86 << 0 << 1.91 << 3.84 << endr;
	hcmc  << 150   << 107.5 << 89.8 << 76.5 << 59.6 << endr;
	hthe  << 0	   << -10 	<< -40  << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
		  << 9     << 6 	<< 1    << 9 << 8 << 7  << 4 << 8 << 7 << 6 << 2
		  << 7 	   << 7 	<< 7    << endr;

	string path  = "misc";
	string file1 = path + "/hgeo.dat"; // unit = mm
	string file2 = path + "/rad.dat";
	hgeo.load(file1, raw_ascii);
	hrad.load(file2, raw_ascii);

	hrad = hrad / 10.; // convert to cm
	hgeo = hgeo / 10.; // convert to cm

	mat cen = zeros<mat>(48,3);

	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad);
	hand.build_hand_model(hthe, cen);

	return (hand);
}


int main() {
	/* code */

	// set armadillo rng seed and print in scientific format
	arma_rng::set_seed(10000);
	cout.precision(15);
	cout << scientific << endl;

	int test_no = 2;

	if (test_no == 0) {
		observedmodel observation;
		test_observedmodel(observation, true); // downsample == true

		handmodel hand = test_handmodel();
		costfunc optf(&hand, &observation);
	}

	else if (test_no == 1) {
		test_visualiser();
	}

	else if (test_no == 2) {
		time_t start = clock();
		test_full();
		cout << "execution time = " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;
	}

	else {
		cout << "invalid arg" << endl;
	}

	return (0);
}

