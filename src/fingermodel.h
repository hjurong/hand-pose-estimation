#ifndef FINGERMODEL_H
#define FINGERMODEL_H

#include <armadillo>

using namespace arma;

class fingermodel {
public:
	fingermodel();

	vec get_f_geometry() const;
	vec get_num_spheres() const;
	float get_spacing() const;
	double get_CMC() const;

	void init(double tCMC, vec finger_geo, float spacing, vec numSpheres);
	void set_CMC(double CMC);
	void set_spacing(float nspacing);
	void set_f_geometry(vec finger_geo);
	void set_num_spheres(vec numSpheres);
	void set_transform_mat(mat &T12, mat &T23, mat &T34, 
						   mat &T45, mat &T00, mat &Tgb, 
						   vec &f_geometry,
						   vec &gb_trans,
						   vec &g_pos,
						   vec &theta);

	void buildSpheres(mat &joints, mat &sphere_centres);
	void build_finger_model(vec &f_theta, vec &gb_trans, 
						    vec &g_pos, mat &sphere_centres);

	mat::fixed<5,3> finger_joints;

private:

	// T01 and T10 are transformation matrices relating to CMC
	// they do not change with theta
	mat::fixed<4,4> T01, T10;
	vec::fixed<4> f_geometry; // finger geometry; i.e. length between two joints
	vec::fixed<4> num_spheres; // number of spheres between adjacent joints

	float spacing;
	double CMC;
	bool setCMCTrans;

	// private functions
	double deg2rad(double angle_deg);
	
};



#endif
