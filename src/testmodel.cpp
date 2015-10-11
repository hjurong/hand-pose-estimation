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

	vec x0(26), hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);
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

	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad); // initialise hand

	string home = "../";
	string subj = "handModelling/Release_2014_5_28/Subject1/";
	string objt = "000000_depth.bin";
	string full = home + subj;

	double focal = 241.42;
	int imgH = 320, imgW = 240;
	bool conv_to_cm = true;
	bool downsample = true;

	observedmodel observation;
	observation.init_observation(full, objt, conv_to_cm, imgW, imgH,
								 focal, downsample);

	costfunc optfunc(&hand, &observation);

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
	int maxiter = 80;
	int num_p = 32;
	double minstep = 1e-8;
	double minfunc = 1e-8;

	mat gnd_truth_joints;
	gnd_truth_joints.load("t.dat");

	PSO optimiser;
	optimiser.set_pso_params(ub, lb, std, w, c1, c2,
							 maxiter, minstep, minfunc);

	int numframes = 1;
	vec fitness = zeros<vec>(numframes);

//	cv::namedWindow("disttrans", cv::WINDOW_AUTOSIZE ); // Create a window for display.


	for (int frame=0; frame < numframes; frame++) {
		stringstream ss;
		ss << setw(6) << setfill('0') << frame;
		string next_frame_num = ss.str();

		string nextframe_name = next_frame_num + "_depth.bin";

		observation.next_frame(nextframe_name);

//		optimiser.refine_init_pose(x0, optfunc);

		vec bestp = zeros<vec>(26);
//		optimiser.pso_optimise(optfunc, x0, num_p, bestp);

		optimiser.pso_evolve(optfunc, x0, num_p, bestp);

		double c = optfunc.cal_cost(bestp);
		string label = "frame" + next_frame_num + "-cost: ";
		cout << label << c << endl;

		double c2 = optfunc.gnd_truth_err(gnd_truth_joints, frame);
		cout << "error = " << c2 << endl;

		mat::fixed<48,3> spheres;
		vec t = zeros<vec>(26);
		hand.build_hand_model(t, spheres);
		spheres.save("spheres_cano.dat", raw_ascii);
//
//		string name = "hjoints_zeros.dat";
//		mat joints = hand.hand_joints;
//		joints.cols(1,2) *= -1;
//		joints.save(name, raw_ascii);
//
//		cv::Mat out_img = optfunc.depthMatch_penalty(spheres);
//		cv::imshow("disttrans", out_img); // Show our image inside it.
//		cv::waitKey(0);

		fitness(frame) = c2; // store fitness for current frame

		x0 = bestp;

	}

//	fitness.save("fitness_error.dat", raw_ascii);


}

void test_PSO(costfunc &optfunc) {
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
	int maxiter = 80;
	int num_p = 32;
	double minstep = 1e-8;
	double minfunc = 1e-8;

	PSO optimiser;
	optimiser.set_pso_params(ub, lb, std, w, c1, c2,
							 maxiter, minstep, minfunc);

	optimiser.refine_init_pose(x0, optfunc);

	vec bestp = zeros<vec>(26);

//	optimiser.pso_optimise(optfunc, x0, num_p, bestp);

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
	hthe << 0 <<-10<< -40 << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
		 << 9 << 6 << 1  << 9  << 8 << 7 << 4 << 8 << 7 << 6 << 2
		 << 7 << 7 << 7  << endr;

	costfunc costfunction(hand, observation);

//	mat ptncloud, disttrans;
//	observation->depth_to_ptncloud(ptncloud);
//	observation->dist_transform(disttrans);
//
//	double scale = observation->get_img_scale();
//	mat cam_mat = observation->get_cam_mat();
//	mat depthm = observation->get_depth();
//	vec radii = hand->get_spheres_radii();
//	bool comp = true;
//
//	double totalcost = costfunction.cal_cost(hthe, ptncloud, radii,
//								disttrans, depthm, cam_mat, scale, comp);
//
//	cout << totalcost << endl;
//
//	observation->next_frame("000001_depth.bin");

	mat::fixed<48,3> spheresM;
	hand->build_hand_model(hthe, spheresM);
//	double c = costfunction.bincomp_penalty(spheresM);
//	cout << c << endl;

//	double c2 = costfunction.cal_cost(hthe);
//	cout << c2 << endl;


//	vec &theta, mat &ptncloud, vec &radii, mat &disttrans,
//	mat &depthmap, mat &cam_mat, double scale

//	double n = costfunction.align_models(false);
//	cout <<"align-cost: " << n << endl;
//
//	double c = costfunction.depth_penalty();
//	cout << "depth-penalty: " << c << endl;
//
//	double l = costfunction.self_collision_penalty();
//	cout << "collision penalty: " << l << endl;

//	hthe.row(4) -= 12;
//	cout << hthe << endl;



}


void test_observedmodel(observedmodel &observation, bool downsample) {
	string home = "../";
	string path = "handModelling/Release_2014_5_28/Subject1/";
	string objt = "000001_depth.bin";
	string full = home + path;

	double focal = 241.42;
	int imgH = 320, imgW = 240;
	bool conv_to_cm = true;
	mat ptncloud;
//	observedmodel observation;
	observation.init_observation(full, objt, conv_to_cm, imgW, imgH,
								 focal, downsample); // already calls depth_to_ptncloud


//	uvec id;
//	observation.downsample_ptncloud(id);
//	observation.depth_to_ptncloud(ptncloud);
//	observation.show_depthmap();
//	observation.dist_transform(ptncloud);

//	double scale = observation.get_img_scale();
//	cout << scale << endl;

}

void test_visualiser() {

//	vec x0(26), hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);
//	tbnum << 2 << 2 << 2 << 2 << endr;
//	fgnum << 4 << 2 << 2 << 2 << endr;
//	spc << -1.86 << -1.86 << 0 << 1.91 << 3.84 << endr; // digit spacing
//	hcmc << 150 << 107.5 << 89.8 << 76.5 << 59.6 << endr; // cmc angles
//	x0 << 0 << -10 << -40 << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
//	   << 9 << 6 << 1  << 9  << 8 << 7 << 4 << 8 << 7 << 6 << 2
//	   << 7 << 7 << 7  << endr; // initial pose
//
//	// load hand geometry
//	string handparam_path = "misc";
//	string file1 = handparam_path + "/hgeo.dat"; // unit = mm
//	string file2 = handparam_path + "/rad.dat";
//	hgeo.load(file1, raw_ascii);
//	hrad.load(file2, raw_ascii);
//
//	hrad = hrad / 10.; // convert to cm
//	hgeo = hgeo / 10.; // convert to cm
//
//	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad); // initialise hand


	gl_visualise();


//	cout << 111 << endl;


}


handmodel test_handmodel() {
	///////////////////////////////////////////////////////////
	//// handmodel testing
	///////////////////////////////////////////////////////////
	vec hthe(26), hgeo(20), tbnum(4), fgnum(4), spc(5), hcmc(5), hrad(48);
	tbnum << 2 << 2 << 2 << 2 << endr;
	fgnum << 4 << 2 << 2 << 2 << endr;
	spc << -1.86 << -1.86 << 0 << 1.91 << 3.84 << endr;
	hcmc << 150 << 107.5 << 89.8 << 76.5 << 59.6 << endr;
	hthe << 0 << -10 << -40 << 0 << 3 << 32 << 6 << 9 << 8 << 9 << 3 << 9
		 << 9 << 6 << 1  << 9  << 8 << 7 << 4 << 8 << 7 << 6 << 2
		 << 7 << 7 << 7  << endr;
//	hgeo << 3.6 << 3.7 << 3.6 << 2.8 << 7.8 << 4.7 << 2.7 << 2.4
//		 << 8.1 << 5.0 << 3.1 << 2.5 << 7.8 << 4.7 << 2.7 << 2.4
//		 << 7.7 << 3.5 << 2.3 << 2.2 << endr;

	string path = "misc";
	string file1 = path + "/hgeo.dat"; // unit = mm
	string file2 = path + "/rad.dat";
	hgeo.load(file1, raw_ascii);
	hrad.load(file2, raw_ascii);

	hrad = hrad / 10.; // convert to cm
	hgeo = hgeo / 10.; // convert to cm

//	cout << hgeo << endl;
//	cout << hrad << endl;
//
//	cout << hrad << "--\n" << endl;

//	<< 2.5 << 1.5 << 1.5 << 1.0 << 5.0 << 2.5 << 2.0 << 1.0
//	<< 5.0 << 3.0 << 2.0 << 1.0 << 5.0 << 2.5 << 2.0 << 1.0
//	<< 5.0 << 1.5 << 1.0 << 0.7 << endr;

	mat cen = zeros<mat>(48,3);

	handmodel hand(hgeo, spc, tbnum, fgnum, hcmc, hrad);
	hand.build_hand_model(hthe, cen);
//
//	cen.print("CEN: ");
//	hand.build_hand_model();
//	hand.build_spheres();
//	cout << hand.get_hand_geo() << endl;
//	cout << hand.get_joint_pos(false) << endl;
//	cout << hand.get_spheres_pos(false) << endl;
//	cout << hand.get_spheres_radii(false) << endl;

	return (hand);
}

int main() {
	/* code */
/////////////////////////////////////////////////////////////////
//// init-params for finger and thumb model
/////////////////////////////////////////////////////////////////

//	vec theta(5), f_geo(4), g_pos(3), gb_trans(2), num_s(4);
//	float spacing = 0.5;
//	double CMC = 45;
//	// test params
//	theta << 10 << 5 << 6 << 5 << endr;
//	f_geo << 2.5 << 1.5 << 1.5 << 1 << endr;
//	g_pos << 5  <<  0 << 5 << endr;
//	gb_trans << 30 << 30 << endr;
//	num_s << 4 << 2 << 2 << 2 << endr;

////////////////////////////////////////////////////////////
//// fingermodel testing
////////////////////////////////////////////////////////////

//	fingermodel index;
//	index.init(theta, CMC, f_geo, g_pos, gb_trans, spacing, num_s);
//	cout << index.getTheta() << endl;
//	index.transform();
//	cout << index.get_joint_pos() << endl;
//	index.buildSpheres();
//	cout << index.get_spheres_pos() << endl;
//	cout << index.get_sphere_radius() << endl;
	//	cout << index.get_joint_pos() << endl;
	//	mat::fixed<6,3> abc = index.get_joint_pos();
	//	abc.print("ABC: ");

//////////////////////////////////////////////////////////////
//// thumbmodel testing
//////////////////////////////////////////////////////////////

//	thumbmodel thumb;
//
//	thumb.init(theta, CMC, f_geo, g_pos, gb_trans, spacing, num_s);
//	cout << thumb.getTheta() << endl;
//	thumb.transform();
//	cout << thumb.get_joint_pos() << endl;
//	thumb.buildSpheres();
//	cout << thumb.get_spheres_pos() << endl;
//	cout << thumb.get_sphere_radius() << endl;

//////////////////////////////////////////////////////////////////
//	loadbin();

//////////////////////////////////////////////////////////////////

	arma_rng::set_seed(10000);
	cout.precision(15);
	cout << scientific << endl;

///////////////////////////////////////////////////////////////////
// test 1
///////////////////////////////////////////////////////////////////
//	observedmodel observation;
//	test_observedmodel(observation, true); // downsample == true
//
////	mat ptncld = *(observation.get_ptncloud());
////	ptncld.save("ptncld.dat", raw_ascii);
//
//	handmodel hand = test_handmodel();
////
//	test_cost(&hand, &observation);
////
//	costfunc optf(&hand, &observation);
//
//	cout << "start PSO" << endl;
//	time_t start = clock();
//	test_PSO(optf);
//
//
//	cout << "execution time = " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;
///////////////////////////////////////////////////////////////////////////////////

// test visualiser
//	test_visualiser();

///////////////////////////////////////////////////////////////////////////////////
//// full test
	time_t start = clock();
	test_full();
	cout << "execution time = " << (double)(clock()-start)/CLOCKS_PER_SEC << endl;


	return (0);
}

