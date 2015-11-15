#include "handmodel.h"
#include "fingermodel.h"
#include "thumbmodel.h"
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

handmodel::handmodel(vec h_geo, vec h_spacing,
					 vec tb_spheres, vec fg_spheres, vec h_CMC, vec sphR) {
	/*
	 * initialise hand model; setting parameters
	 */

	set_hand_CMC(h_CMC);
	set_hand_geo(h_geo);
	set_spacing(h_spacing);
	set_num_spheres(tb_spheres, fg_spheres);
	set_hand_rad(sphR);

	// instantiate 4 finger models and 1 thumb model after setting the parameters
	init_hand();

	this->hand_joints = zeros<mat>(21,3); // initialise to zeros

}


void handmodel::update_params(bool cmc = false,
							  bool geo = false,
							  bool spacing = false) {
	/* inputs: booleans ==> if TRUE, update parameters
	 *
	 * update the corresponding parameters if set to TRUE
	 */

	if (cmc) {
		thumb.set_CMC(this->hand_CMC(0));
		index.set_CMC(this->hand_CMC(1));
		middle.set_CMC(this->hand_CMC(2));
		ring.set_CMC(this->hand_CMC(3));
		little.set_CMC(this->hand_CMC(4));

	}

	if (geo) {
		vec::fixed<4> geo_array[5];
		process_geo(geo_array);

		thumb.set_tb_geometry(geo_array[0]);
		index.set_f_geometry(geo_array[1]);
		middle.set_f_geometry(geo_array[2]);
		ring.set_f_geometry(geo_array[3]);
		little.set_f_geometry(geo_array[4]);

	}

	if (spacing) {
		thumb.set_spacing(this->spacing(0));
		index.set_spacing(this->spacing(1));
		middle.set_spacing(this->spacing(2));
		ring.set_spacing(this->spacing(3));
		little.set_spacing(this->spacing(4));
	}

}

void handmodel::init_hand() {
	/*
	 * initialse the fingers (i.e. 4 finger models and 1 thumb model)
	 *
	 */
	vec geo_array[5]; // = {tb_geo, f1_geo, f2_geo, f3_geo, f4_geo};

	process_geo(geo_array); // split the input hand geometry vector

	// initialise the finger and thumb models
	this->thumb.init(hand_CMC(0), geo_array[0], spacing(0), tb_num_spheres);
	this->index.init(hand_CMC(1), geo_array[1], spacing(1), fg_num_shperes);
	this->middle.init(hand_CMC(2), geo_array[2], spacing(2), fg_num_shperes);
	this->ring.init(hand_CMC(3), geo_array[3], spacing(3), fg_num_shperes);
	this->little.init(hand_CMC(4), geo_array[4], spacing(4), fg_num_shperes);

}


void handmodel::process_rad(vec &tb_radii, vec rad_array[]) {
	/* input:  arma::vec = the spheres radii vector
	 * output: an array of arma:vec containing the spheres radii corresponding
	 * 		   to each finger
	 */

	int base, count = 4;
	int start = sum(this->tb_num_spheres), end = 0;
	int incre = sum(this->fg_num_shperes);
	tb_radii = this->spheres_radii(span(0, start-1));

	for (int i = 0; i < count; ++i) {
		base = i*incre + start;
		end = base + incre-1;
		rad_array[i] = this->spheres_radii(span(base, end));

	}
}

void handmodel::process_geo(vec geo_array[]) {
	/*
	 * split the hand geometry vector into an array of arma:vec,
	 * corresponding to the joint lengths for each finger
	 *
	 */
	int count = 5;
	int start, end;
	for (int i = 0; i < count; ++i) {
		/* code */
		start = i*4;
		end = (i+1)*4-1;
		geo_array[i] = this->hand_geo(span(start, end));
	}
}

void handmodel::process_theta(vec &hand_theta, vec the_array[],
							  vec & g_pos, vec & gb_trans) {
	/*
	 * split the input 26-dim joint angles theta to:
	 *
	 * 1. global rotation
	 * 2. global translation
	 * 3. joint angles for each digit
	 *
	 */

	int count = 5;
	int start;
	int end;

	g_pos    = hand_theta(span(3,5)); // 3 global position parameters
	gb_trans = hand_theta(span(0,2)); // first 3 global rotation parameters

	for (int i = 0; i < count; ++i) {
		/* code */
		start = (i)*4+6; // start inclusive
		end   = (i+1)*4+5; // end inclusive
		the_array[i] = hand_theta(span(start, end));
	}


}

void handmodel::set_spacing(vec &h_spacing) {
	/*
	 * h_spacing specifies the spacing between neighbouring base joints;
	 * this allows the user to manually decide the width of the base (i.e. wrist);
	 * overwriting the sphere radii
	 */
	if (h_spacing.n_rows != 5) {

		spacing.fill(0.0);

		cout << "h_spacing Specifies the distance between neighbouring base joints; ";
		cout << "must be of shape (5,1)" << endl;
		cout << "call set_spacing() to re-initialise" << endl;
	}
	else {
		this->spacing = h_spacing;

	}
}

void handmodel::set_hand_geo(vec &h_geo) {
	/* code
	 * h_geo contains the geometry of each finger/thumb (each with four segments)
	 *
	 * [0:4] = thumb_geo
	 * [4:8] = index_geo ... etc.
	 */
	if (h_geo.n_rows != 20) {

		hand_geo.fill(0.0);

		cout << "hand_geo contains the geometry of each finger/thumb; ";
		cout << "each instance has four segments ==> 20 params are needed ";
		cout << "call set_hand_geo() to re-initialise" << endl;
	}
	else {
		this->hand_geo = h_geo;

	}
}

void handmodel::set_hand_CMC(vec &h_CMC) {
	/* code
	 * h_CMC contains the CMC angel for each finger/thumb
	 * ==> must be of length 5
	 */
	if (h_CMC.n_rows != 5) {

		hand_CMC.fill(0);

		cout << "hand_CMC contains the CMC-angels for each finger/thumb";
		cout << "==> must be of size (5, 1)" << endl;
		cout << "call set_hand_CMC() to re-initialise" << endl;
	}
	else {
		this->hand_CMC = h_CMC;

	}
}

void handmodel::set_num_spheres(vec &tb_spheres, vec &fg_spheres) {
	/*
	 * predefine the number of spheres to be fitted on each finger
	 */
	this->tb_num_spheres = tb_spheres;
	this->fg_num_shperes = fg_spheres;
}

void handmodel::set_hand_rad(vec &rad) {
	this->spheres_radii = rad;
}


/*
 * getter functions
 *
 * */

vec handmodel::get_hand_CMC() const {
	return (this->hand_CMC);
}

vec handmodel::get_hand_geo() const {
	return (this->hand_geo);
}

vec handmodel::get_spacing() const {
	return (this->spacing);
}

vec handmodel::get_tb_num_spheres() const {
	return (this->tb_num_spheres);
}

vec handmodel::get_fg_num_spheres() const {
	return (this->fg_num_shperes);
}

vec handmodel::get_spheres_radii() const {

	return (this->spheres_radii);
}

vec* handmodel::get_radii() {
	return (&this->spheres_radii);
}


void handmodel::build_hand_model(vec &h_theta, mat &sphere_centres) {
	/* input:  the 26-dim theta representing the hand joints
	 * output: the hand spheres model; sphere_centres.shape == (48, 3)
	 *
	 * this relies on the child classes' build_model methods
	 *
	 * */
	vec::fixed<3> g_pos; // elem 3 -- 5 in h_theta
	vec::fixed<3> gb_trans; // elem 0 -- 2 in h_theta
	vec the_array[5]; // theta array == [tb, id, md, rg, lt]

	process_theta(h_theta, the_array, g_pos, gb_trans);

	mat::fixed<8,3> tb_sphere_pos;
	mat::fixed<10,3> id_sphere_pos, md_sphere_pos;
	mat::fixed<10,3> rg_sphere_pos, lt_sphere_pos;

	thumb.build_thumb_model(the_array[0], gb_trans, g_pos, tb_sphere_pos);
	index.build_finger_model(the_array[1], gb_trans, g_pos, id_sphere_pos);
	middle.build_finger_model(the_array[2], gb_trans, g_pos, md_sphere_pos);
	ring.build_finger_model(the_array[3], gb_trans, g_pos, rg_sphere_pos);
	little.build_finger_model(the_array[4], gb_trans, g_pos, lt_sphere_pos);

	sphere_centres.rows(0,7) = tb_sphere_pos;
	sphere_centres.rows(8, 17) = id_sphere_pos;
	sphere_centres.rows(18, 27) = md_sphere_pos;
	sphere_centres.rows(28, 37) = rg_sphere_pos;
	sphere_centres.rows(38, 47) = lt_sphere_pos;

	sphere_centres.cols(1,2) *= -1; // ensure consistent coordinate system

	// store hand joint positions
	this->hand_joints.row(0) 	  = g_pos.t();
	this->hand_joints.rows(1,4)   = index.finger_joints.rows(1,4);
	this->hand_joints.rows(5,8)   = middle.finger_joints.rows(1,4);
	this->hand_joints.rows(9,12)  = ring.finger_joints.rows(1,4);
	this->hand_joints.rows(13,16) = little.finger_joints.rows(1,4);
	this->hand_joints.rows(17,20) = thumb.thumb_joints.rows(1,4);

}
