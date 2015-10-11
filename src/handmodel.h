#ifndef HANDMODEL_H
#define HANDMODEL_H

#include "fingermodel.h"
#include "thumbmodel.h"

class handmodel {
public:
	handmodel(vec h_geo, vec h_spacing,
			  vec tb_spheres, vec fg_spheres, vec h_CMC, vec sphR);

	void init_hand();
	void set_hand_CMC(vec &h_CMC);
	void set_hand_geo(vec &h_geo);
	void set_spacing(vec &h_spacing);
	void set_hand_rad(vec &rad);
	void set_num_spheres(vec &tb_spheres, vec &fg_spheres);
	void gl_visualise();

	vec get_hand_geo() const;
	vec get_hand_CMC() const;
	vec get_spacing() const;
	vec get_tb_num_spheres() const;
	vec get_fg_num_spheres() const;
	vec get_spheres_radii() const;

	vec* get_radii();

	handmodel & update_hand();
	handmodel & build_spheres();

	void build_hand_model(vec &h_theta, mat &sphere_centres);

	mat hand_joints;

private:

	fingermodel index, middle, ring, little;
	thumbmodel thumb;

	vec::fixed<20> hand_geo;
	vec::fixed<5> spacing;
	vec::fixed<5> hand_CMC;
	vec::fixed<4> tb_num_spheres;
	vec::fixed<4> fg_num_shperes;

	vec spheres_radii;


	void process_theta(vec &hand_theta, vec the_array[], vec &g_pos, vec &gb_trans);
	void process_geo(vec geo_array[]);
	void process_rad(vec &tb_radii, vec rad_array[]);
	void update_params(bool cmc, bool geo, bool spacing);
	void update_theta();

};

#endif
