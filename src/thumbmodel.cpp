#include "thumbmodel.h"
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

const static double PI = acos(-1);

thumbmodel::thumbmodel() {
	this->spacing = 0.0;
	this->CMC = 0.0;
	this->Trf.fill(0.0);
	this->T10.fill(0.0);
	this->thumb_joints.fill(0.0);
	this->setCMCTrans = true;
}


//thumbmodel::thumbmodel(vec Thetas, vec tb_geo, vec global_pos, vec global_trans,
//					   float dis2gpos, vec numSpheres) {
//	spacing = dis2gpos;
//	joint_pos.fill(0.0);
//
//	// call setters and check inputs
//	setTheta(Thetas);
//	set_tb_geometry(tb_geo);
//	set_g_pos(global_pos);
//
//	// setting num_spheres also initialise mat-->spheres_pos & vec-->sphere_radius
//	set_num_spheres(numSpheres);
//	set_gb_trans(global_trans); // global rotational transforms
//
//	// initialise ALL (i.e. set_all==true) transformation matrices
//	set_transform_mat(true);
//
//}

void thumbmodel::init(double tCMC, vec tb_geo, float dis2gpos,
					  vec numSpheres) {


	// call setters and check inputs
	set_spacing(dis2gpos);
	set_CMC(tCMC);
	set_tb_geometry(tb_geo);

	// setting num_spheres also initialise mat-->spheres_pos & vec-->sphere_radius
	set_num_spheres(numSpheres);

}

/*
setter functions
*/

void thumbmodel::set_spacing(float nspacing) {
	this->spacing = nspacing;
}

void thumbmodel::set_CMC(double tCMC) {
	this->CMC = tCMC;
	this->setCMCTrans = true;
}

//void thumbmodel::set_sphere_radius(vec sradius) {
//	sphere_radius = sradius;
//}

void thumbmodel::set_num_spheres(vec numSpheres) {
	if (numSpheres.n_rows != 4) {
		/* code */
		num_spheres.fill(0.0);
		cout << "size of numSpheres must be (4,1)" << endl;
		cout << "call set_num_spheres() to reset the params" << endl;
	} else {
		this->num_spheres = numSpheres;
	}
}

void thumbmodel::set_tb_geometry(vec tb_geo) {
	if (tb_geo.n_rows != 4 || tb_geo.n_cols != 1) {
		/* code */
		tb_geo.fill(0.0);
		cout << "size of tb_geometry must be (4,1)" << endl;
		cout << "call set_tb_geometry() to reset the params" << endl;
	}
	else {
		this->tb_geometry = tb_geo;
	}
}


void thumbmodel::set_transform_mat(mat &T01, mat &T12, mat &T23, 
								   mat &T34, mat &T00, mat &Tgb,
								   vec &tb_geometry, vec &gb_trans, 
								   vec &g_pos, vec &theta) {
	double nCMC, TMC1, TMC2, MCP, IP;
	nCMC = deg2rad(this->CMC);
	TMC1 = deg2rad(theta(0));
	TMC2 = deg2rad(theta(1));
	MCP  = deg2rad(theta(2));
	IP   = deg2rad(theta(3));

	double TWS, ANG, ROT; // rotation in z, y, x
	TWS = deg2rad(gb_trans(0) + 180); // y axis is inverted ==> need to roated z-axis by 180deg
	ANG = deg2rad(gb_trans(1));
	ROT = deg2rad(gb_trans(2));

	double L0, L1, L2, L3;
	L0 = tb_geometry(0);
	L1 = tb_geometry(1);
	L2 = tb_geometry(2);
	L3 = tb_geometry(3);

	double ux, uy, uz;
	ux = g_pos(0);
	uy = g_pos(1);
	uz = g_pos(2);


	if (this->setCMCTrans) {

		
		/* code
		CMC is fixed; thus its transformation does not need to be
		updated everytime
		*/
		this->Trf << cos(nCMC) << -sin(nCMC) << 0 << L0*cos(nCMC) << endr
			  	  << sin(nCMC) <<  cos(nCMC) << 0 << L0*sin(nCMC) << endr
				  << 		 0 << 		   0 << 1 << 	    	0 << endr
			 	  << 		 0 << 	       0 << 0 <<		   	1 << endr;

		/*
		T10 is the reverse of Trf so that the spheres on the
		palm can be generated properly
		trigonomatry is used to enasure that the distance between
		every two neighbouring spheres == spacing
		*/
		double a = sqrt(L0*L0+spacing*spacing-2*L0*spacing*cos(nCMC));
		double beta = asin(sin(nCMC)*spacing/a); // rotation angle

//		cout << a << endl;
//		cout << beta << endl;

		this->T10 << cos(beta) << -sin(beta) << 0 << -a*cos(beta) << endr
				  << sin(beta) <<  cos(beta) << 0 << -a*sin(beta) << endr
				  << 		 0 << 		   0 << 1 << 		    0 << endr
				  << 		 0 << 		   0 << 0 <<		   	1 << endr;

		this->setCMCTrans = false;
	}


	/*
	remaing transformations for the rest of the joints are updated every time
	the DH-param can be determined by looking at:
	T**(2,1) ~ 0, -1, 1
	*/
	T01 << cos(TMC1) <<  0 << -sin(TMC1) << 0 << endr
		<< sin(TMC1) <<  0 <<  cos(TMC1) << 0 << endr
		<<		   0 << -1 << 		   0 << 0 << endr
		<< 		   0 <<  0 << 		   0 << 1 << endr;

	double pCMC = nCMC + PI;
	T12 << cos(TMC2) << -sin(TMC2)*cos(pCMC) <<  sin(TMC2)*sin(pCMC) << L1*cos(TMC2) << endr
		<< sin(TMC2) <<  cos(TMC2)*cos(pCMC) << -cos(TMC2)*sin(pCMC) << L1*sin(TMC2) << endr
		<< 		   0 << 		   sin(pCMC) << 		   cos(pCMC) << 		   0 << endr
		<< 		   0 << 		 		   0 << 				   0 << 		   1 << endr;

	T23 << cos(MCP) << -sin(MCP) << 0 << L2*cos(MCP) << endr
		<< sin(MCP) <<  cos(MCP) << 0 << L2*sin(MCP) << endr
		<< 		  0 << 		   0 << 1 << 		   0 << endr
		<<		  0 << 		   0 << 0 << 		   1 << endr;

	T34 << cos(IP) << -sin(IP) << 0 << L3*cos(IP) << endr
		<< sin(IP) <<  cos(IP) << 0 << L3*sin(IP) << endr
		<<		 0 << 		 0 << 1 << 		    0 << endr
		<< 		 0 << 	     0 << 0 << 		    1 << endr;

	/*
	global position --> specifies movements in 3D
	*/
	T00 << 1 << 0 << 0 << ux << endr
		<< 0 << 1 << 0 << uy << endr
		<< 0 << 0 << 1 << uz << endr
		<< 0 << 0 << 0 <<  1 << endr;

	/*
	global transformation matrix for rotations --> hand rotations in 3D
	*/
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
//	mat::fixed<4,4> test[8] = {Trf, T01, T10, T12, T23, T34, T45, Tgb};
//	for (int i = 0; i<8; ++i) {
//		test[i].print("\nMAT: ");
//	}

}

/*
getter functions
*/

vec thumbmodel::get_tb_geometry() const {
	return (this->tb_geometry);
}

vec thumbmodel::get_num_spheres() const {
	return (this->num_spheres);
}

double thumbmodel::get_CMC() const {
	return (this->CMC);
}

float thumbmodel::get_spacing() const {
	return (this->spacing);
}

/*
private functions
*/
double thumbmodel::deg2rad(double angle_deg) {
	return (angle_deg/180.0*PI);
}

/*
operations on thumbmodel instances
*/
void thumbmodel::buildSpheres(mat &joint_pos, mat &spheres_pos) {

	// spheres_pos.shape = (num_tb_spheres, 3)

	// number of joints -- including finger tips; 
	// recall size(joint_pos) == fixed<5,3>
	int num_joints = 5; 
	int numSpheres = 0;
	int cnt = 0; // counter for current position of sphere_pos
	double t = 0;

	rowvec::fixed<3> sph_cen;
	rowvec::fixed<3> joint1, joint2; // rowvec for position of two neighbouring joints
	

	for (int i = 0; i < num_joints-1; ++i) {
		/* code 
		5 joints ==> 4 segments/parts ==> 4 iterations
		*/

		joint1 = joint_pos.row(i);
		joint2 = joint_pos.row(i+1);

		numSpheres = this->num_spheres(i); // spheres in current segment
		t = 1./(this->num_spheres(i)); // refer to general midpoint formula

		for (int j = 1; j < numSpheres+1; ++j) {
			/* code
			num_spheres(vec): specifies the number of spheres
			in each finger segment

			radii of spheres in one finger segment are equal

			int-j must start at 1 to ensure int-mapping starts at 1
			*/

			// apply general midpoint formula to get the next sphere center
			// add the positions to spheres_pos
			// and radii to sphere_radius
			sph_cen = (1.-t*j)*joint1 + t*j*joint2;
			spheres_pos.row(cnt) = sph_cen;

			cnt += 1;
		}

	}

}

void thumbmodel::build_thumb_model(vec &tb_theta, vec &gb_trans, vec &g_pos,
						   		   mat &sphere_centres) {
	//	cout << T01 << T12 << endl;
	mat::fixed<4,4> T01, T12, T23, T34, T00, Tgb;

	set_transform_mat(T01, T12, T23, T34, T00, Tgb, this->tb_geometry,
					  gb_trans, g_pos, tb_theta);

	mat::fixed<4,4> T012 = T01 * T12; // T12 and T23 combine to give the next ptn
	mat::fixed<4,4> *T_mat_pointers[4] = {&Trf, &T012, &T23, &T34};

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
			joint_pos.row(0) = temp.rows(0,2).t(); //temp(span(0,2)).t();

		}

		current_pos = current_pos * (*T_mat_pointers[i]); // apply transformation

		// extract the last column, which contains current position of joint
		// in homogeneous coordinates
		// only first 3 elements are appended to rows of joint_pos
		vec::fixed<4> temp = current_pos.col(3);
		joint_pos.row(i+1) = temp.rows(0,2).t(); // temp(span(0,2)).t(); // update joint_pos

	}

	this->thumb_joints = joint_pos;

	buildSpheres(joint_pos, sphere_centres);
}
