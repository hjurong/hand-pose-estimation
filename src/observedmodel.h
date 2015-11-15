#ifndef OBSERVEDMODEL_H
#define OBSERVEDMODEL_H

#include <armadillo>
#include <string>

using namespace arma;
using namespace std;

class observedmodel {

private:

	string path;
	string filename;

	mat depthmap;
	mat pointcld;
	mat disttran;

	vec::fixed<2> img_center;
	mat::fixed<3,3> camera_calibration;

	int imgW;
	int imgH;
	bool to_cm;
	bool downsample;
	double focal_len;
	double scale;


public:

	observedmodel();

	void init_observation(string dpath, string dfname, bool mm_to_cm,
						  int imW, int imH, double foclen, bool downsample);
	void get_observed();
	void show_depthmap();
	void visualise_ptncloud();
	void depth_to_ptncloud(mat &ptncloud);
	void load_data();
	void downsample_ptncloud(uvec &rows_id);

	void set_focal_len(double nfocal);
	void set_img_center(vec ncenter);
	void set_filename(string name);
	void set_path(string newdir);
	void set_mm_to_cm(bool mm_to_cm);

	mat* get_camera_mat();
	mat* get_depthmap();
	mat* get_ptncloud();
	mat* get_disttran();

	mat get_cam_mat() const;
	mat get_depth() const;
	vec get_img_center() const;

	double get_img_scale() const;
	double get_focal_len() const;
	int get_imgH() const;
	int get_imgW() const;

	observedmodel & next_frame(string next);

	void invert_depthmap(mat &depthmp, bool display=false);
	void dist_transform(mat &dist_trans);

};

#endif
