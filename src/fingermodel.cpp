#include "fingermodel.h"
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

const static double PI = acos(-1);

/*
constructor
*/
fingermodel::fingermodel() {
	this->spacing = 0.0;
	this->CMC = 0.0;
	this->T01.fill(0.0);
	this->T10.fill(0.0);
	this->finger_joints.fill(0.0);
	this->setCMCTrans = true;
}


//fingermodel::fingermodel(vec Thetas, vec finger_geo, vec global_pos,
//						 vec global_trans, float dis2gpos,
//						 vec numSpheres) {
//	// initialise joint_positions and spheres_positions to zeros-array
//	spacing = dis2gpos; // needs to be set before calling set_transform_mat()
//	joint_pos.fill(0.0);
//
//	// call setter functions
//	setTheta(Thetas);
//	set_f_geometry(finger_geo);
//	set_g_pos(global_pos);
//
//	// also sets the size of mat-->spheres_pos
//	// and vec --> sphere_radius
//	set_num_spheres(numSpheres);
//	set_gb_trans(global_trans);
//
//	// once the parames (i.e. theta, f_geometry and g_pos) are initialised
//	// generate the appropriate transformation matrices
//	set_transform_mat(true); // set_all = true
//
//}

void fingermodel::init(double tCMC, vec finger_geo, float dis2gpos,
					   vec numSpheres) {
	// initialise joint_positions and spheres_positions to zeros-array

	// call setter functions
	set_spacing(dis2gpos);
	set_CMC(tCMC);
	set_f_geometry(finger_geo);

	// also sets the size of mat-->spheres_pos
	// and vec --> sphere_radius
	set_num_spheres(numSpheres);

}

/*
setters -- check for validity of input data
*/

void fingermodel::set_spacing(float nspacing) {
	this->spacing = nspacing;
}

void fingermodel::set_CMC(double tCMC) {
	this->CMC = tCMC;
	this->setCMCTrans = true;
}


// void fingermodel::set_sphere_radius(vec sradius) {
// 	sphere_radius = sradius;
// }

void fingermodel::set_num_spheres(vec numSpheres) {
	this->num_spheres = numSpheres;
}

void fingermodel::set_f_geometry(vec finger_geo) {
	if (finger_geo.n_rows != 4 || finger_geo.n_cols != 1) {
		/* code */
		f_geometry.fill(0.0);
		cout << "size of f_geometry must be (4,1)" << endl;
		cout << "call set_f_geometry() to reset the params" << endl;

	} else {
		this->f_geometry = finger_geo;
	}
}

void fingermodel::set_transform_mat(mat &T12, mat &T23, mat &T34, 
						   			mat &T45, mat &T00, mat &Tgb, 
						   			vec &f_geometry, vec &gb_trans, 
						   			vec &g_pos, vec &theta) {

	double MCP1, MCP2, PIP, DIP;
	MCP1 = deg2rad(theta(0));
	MCP2 = deg2rad(theta(1));
	PIP  = deg2rad(theta(2));
	DIP  = deg2rad(theta(3));

	double TWS, ANG, ROT;
	TWS = deg2rad(gb_trans(0) + 180); // z
	ANG = deg2rad(gb_trans(1)); // y
	ROT = deg2rad(gb_trans(2)); // x

	double L4, L5, L6, L7;
	L4 = f_geometry(0);
	L5 = f_geometry(1);
	L6 = f_geometry(2);
	L7 = f_geometry(3);

	double ux, uy, uz;
	ux = g_pos(0);
	uy = g_pos(1);
	uz = g_pos(2);

	if (this->setCMCTrans) {
		double nCMC = deg2rad(this->CMC);
		/* code 
		CMC is fixed; thus its transformation does not need to be 
		updated everytime
		*/
		this->T01 << cos(nCMC) << -sin(nCMC) << 0 << L4*cos(nCMC) << endr
				  << sin(nCMC) <<  cos(nCMC) << 0 << L4*sin(nCMC) << endr
				  << 		 0 << 		   0 << 1 << 		    0 << endr
				  << 		 0 << 		   0 << 0 <<		    1 << endr;

		/*
		T01 is the reverse of T01 so that the spheres on the 
		palm can be generated properly
		trigonomatry is used to enasure that the distance between 
		every two neighbouring spheres == spacing
		*/
		double a = sqrt(L4*L4+spacing*spacing-2*L4*spacing*cos(nCMC));
		double beta = asin(sin(nCMC)*spacing/a); // rotation angle

//		cout << a <<"-a-"<< L4 <<"-L4-" << spacing<<"-sp-" << nCMC << "--cmc"<< endl;
//		cout << f_geometry << endl;
//		cout << theta << "--theta--" << endl;
//		cout << beta << "--beta, in fingermodel; set_transform"<< endl;

		this->T10 << cos(beta) << -sin(beta) << 0 << -L4*sin(nCMC)*cos(beta) << endr
			      << sin(beta) <<  cos(beta) << 0 << -L4*sin(nCMC)*sin(beta) << endr
			      << 		 0 << 		   0 << 1 << 		  			   0 << endr
				  << 		 0 << 		   0 << 0 <<		   			   1 << endr;

		this->setCMCTrans = false; // no need to set CMC transformation matrix again
	}


	T12 << cos(MCP1) <<  0 << -sin(MCP1) << 0 << endr
		<< sin(MCP1) <<  0 <<  cos(MCP1) << 0 << endr
		<<		   0 << -1 << 		   0 << 0 << endr
		<< 		   0 <<  0 << 		   0 << 1 << endr;

	T23 << cos(MCP2) << -sin(MCP2) << 0 << L5*cos(MCP2) << endr
		<< sin(MCP2) <<  cos(MCP2) << 0 << L5*sin(MCP2) << endr
		<< 		   0 << 		 0 << 1 << 			  0 << endr
		<< 		   0 << 		 0 << 0 << 			  1 << endr;

	T34 << cos(PIP) << -sin(PIP) << 0 << L6*cos(PIP) << endr
		<< sin(PIP) <<  cos(PIP) << 0 << L6*sin(PIP) << endr
		<< 		  0 << 		   0 << 1 << 		   0 << endr
		<<		  0 << 		   0 << 0 << 		   1 << endr;

	T45 << cos(DIP) << -sin(DIP) << 0 << L7*cos(DIP) << endr
		<< sin(DIP) <<  cos(DIP) << 0 << L7*sin(DIP) << endr
		<<		  0 << 		   0 << 1 << 		   0 << endr
		<< 		  0 << 		   0 << 0 << 		   1 << endr;

	T00 << 1 << 0 << 0 << ux << endr
		<< 0 << 1 << 0 << uy << endr
		<< 0 << 0 << 1 << uz << endr
		<< 0 << 0 << 0 <<  1 << endr;


	mat Rx, Ry, Rz; // rotation matrix for each axis

	Rz << cos(TWS) << -sin(TWS) << 0 << 0 << endr
	   << sin(TWS) <<  cos(TWS) << 0 << 0 << endr
	   <<		 0 <<		  0 << 1 << 0 << endr
	   <<		 0 << 		  0 << 0 << 1 << endr;

	Ry << cos(ANG) << 0 << sin(ANG) << 0 << endr
	   << 		 0 << 1 << 	  	  0 << 0 << endr
	   <<-sin(ANG) << 0 << cos(ANG) << 0 << endr
	   << 		 0 << 0 << 		  0 << 1 << endr;

	Rx << 1 << 		  0 << 		   0 << 0 << endr
	   << 0 << cos(ROT) << -sin(ROT) << 0 << endr
	   << 0 << sin(ROT) <<  cos(ROT) << 0 << endr
	   << 0 << 		  0 << 		   0 << 1 << endr;

	Tgb = Rz * Ry * Rx;

//	Tgb << cos(TWS) << -sin(TWS)*cos(JNT) << sin(TWS)*sin(JNT) << 0 << endr
//		<< sin(TWS) <<  cos(TWS)*cos(JNT) <<-cos(TWS)*sin(JNT) << 0 << endr
//		<<		   0 << 		 sin(JNT) << 		  cos(JNT) << 0 << endr
//		<< 		   0 <<  				0 <<	   		     0 << 1 << endr;


//	// debug print
//	mat::fixed<4,4> test[8] = {T00, T01, T10, T12, T23, T34, T45, Tgb};
//	for (int i = 0; i<8; ++i) {
//		test[i].print("\nMAT: ");
//	}

}

/*
getters -- return the current value of instance attributes
*/

vec fingermodel::get_f_geometry() const {
	return (f_geometry);
}

vec fingermodel::get_num_spheres() const {
	return (num_spheres);
}

double fingermodel::get_CMC() const {
	return (this->CMC);
}

/*
private functions
*/
double fingermodel::deg2rad(double angle_deg) {
	return (angle_deg/180.0*PI);
}

// /*
// operations on fingermodel instance
// */
// fingermodel& fingermodel::transform() {
// //	cout << T01 << T12 << endl;
// 	mat::fixed<4,4> T123 = T12 * T23; // T12 and T23 combine to give the next ptn
// 	mat::fixed<4,4> T_mat[4] = {T01, T123, T34, T45};

// 	mat::fixed<4,4> current_pos;
// 	current_pos = T00*Tgb;

// 	vec::fixed<4> temp0, temp1;

// 	int i, count = 4;
// 	for (i = 0; i < count; ++i)
// 	{
// 		/* code */
// //		// debug print
// //		cout << T_mat[i] << endl;
// //		current_pos.print("current pos: ");
// //		temp0.print("temp: ");

// 		if (i == 1) {
// 			/* code */
// 			mat::fixed<4,4> base_ptn = current_pos * T10;
// 			temp1 = base_ptn.col(3);
// 			joint_pos.row(0) = temp1(span(0,2)).t();
// 		}

// 		current_pos = current_pos * T_mat[i]; // apply transformation

// 		// extract the last column, which contains current position of joint
// 		// in homogeneous coordinates
// 		// only first 3 elements are appended to rows of joint_pos
// 		temp0 = current_pos.col(3);
// 		joint_pos.row(i+1) = temp0(span(0,2)).t(); // update joint_pos

// 	}

// //	cout << joint_pos << endl;

// 	return (*this);
// }


void fingermodel::buildSpheres(mat &joints, mat &spheres_pos) {
	
	int num_joints = 5; // number of joints; recall size(joint_pos) = fixed::<5,3>
	int numSpheres = 0;
	int cnt = 0; // counter for current position of sphere_pos
	double t = 0;
	rowvec::fixed<3> sph_cen;

	for (int i = 0; i < num_joints-1; ++i) {
		/* code 
		5 joints ==> 4 segments/parts
		*/

		rowvec::fixed<3> joint1 = joints.row(i);
		rowvec::fixed<3> joint2 = joints.row(i+1);

		if (i == 0) {
			numSpheres = this->num_spheres(i);
			t = 1./(this->num_spheres(i)-1);

			for (int j = 0; j < numSpheres; ++j) {
				/* code
				num_spheres(vec): specifies the number of spheres in
				each finger segment

				radii of spheres in one finger segment are equal

				int-j must start at 1 to ensure int-mapping starts at 1
				*/

				sph_cen = (1.-t*j)*joint1 + t*j*joint2;
				spheres_pos.row(cnt) = sph_cen;

				cnt += 1;

			}
		}
		else {
			numSpheres = this->num_spheres(i) + 1;
			t = 1./(this->num_spheres(i));

			for (int j = 1; j < numSpheres; ++j) {
				/* code
				num_spheres(vec): specifies the number of spheres in each finger segment

				radii of spheres in one finger segment are equal

				int-j must start at 1 to ensure int-mapping starts at 1
				*/

				sph_cen = (1.-t*j)*joint1 + t*j*joint2;
				spheres_pos.row(cnt) = sph_cen;

				cnt += 1;

			}
		}
	}

}


void fingermodel::build_finger_model(vec &f_theta, vec &gb_trans, 
									 vec &g_pos, mat &sphere_centres) {
	//	cout << T01 << T12 << endl;
	mat::fixed<4,4> T12, T23, T34, T45, T00, Tgb;

	set_transform_mat(T12, T23, T34, T45, T00, Tgb, this->f_geometry, 
					  gb_trans, g_pos, f_theta);

	mat::fixed<4,4> T123 = T12 * T23; // T12 and T23 combine to give the next ptn
	mat::fixed<4,4> *T_mat_pointers[4] = {&T01, &T123, &T34, &T45};

	mat::fixed<4,4> current_pos = T00*Tgb;

	mat::fixed<5,3> joint_pos;
	
	for (int i = 0; i < 4; ++i) {
		/* code */
//		// debug print
//		cout << T_mat[i] << endl;
//		current_pos.print("current pos: ");
//		temp0.print("temp: ");

		if (i == 1) {
			/* code */
			mat::fixed<4,4> base_ptn = current_pos * T10;
			vec::fixed<4> temp = base_ptn.col(3);
			joint_pos.row(0) = temp.rows(0,2).t();
		}

		current_pos = current_pos * (*T_mat_pointers[i]); // apply transformation

		// extract the last column, which contains current position of joint
		// in homogeneous coordinates
		// only first 3 elements are appended to rows of joint_pos
		vec::fixed<4> temp = current_pos.col(3);
		joint_pos.row(i+1) = temp.rows(0,2).t();
	}

	this->finger_joints = joint_pos;

	buildSpheres(joint_pos, sphere_centres);

}
