#ifndef THUMBMODEL_H
#define THUMBMODEL_H

#include <armadillo>

using namespace arma;

class thumbmodel {
public:
	thumbmodel();
//	thumbmodel(vec Thetas, vec tb_geo, vec global_pos, vec global_trans,
//			   float dis2gpos, vec numSpheres);


	vec get_tb_geometry() const;
	vec get_num_spheres() const;
	float get_spacing() const;
	double get_CMC() const;

	void init(double tCMC, vec tb_geo, float dis2gpos, vec numSpheres);
	void set_CMC(double CMC);
	void set_spacing(float nspacing);
	void set_tb_geometry(vec ntb_geo);
	void set_num_spheres(vec numSpheres);
	void set_transform_mat(mat &T01, mat &T12, mat &T23, 
						   mat &T34, mat &T00, mat &Tgb,
						   vec &tb_geometry, vec &gb_trans, 
						   vec &g_pos, vec &theta);


	// thumbmodel & transform();
	// thumbmodel & buildSpheres();

	void buildSpheres(mat &joints, mat &sphere_centres);
	void build_thumb_model(vec &tb_theta, vec &gb_trans, vec &g_pos,
						   mat &sphere_centres);

	mat::fixed<5,3> thumb_joints;

private:
	mat::fixed<4,4> Trf, T10;
	vec::fixed<4> tb_geometry;
	vec::fixed<4> num_spheres; // 4 joints ==> 4 segments, each with n spheres

	double deg2rad(double angle_deg);

	float spacing;
	double CMC;
	bool setCMCTrans;


};

#endif
