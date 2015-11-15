#include "observedmodel.h"
#include <sys/stat.h>
#include <stdlib.h>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace arma;
using namespace std;

observedmodel::observedmodel() {
	/*
	 * default constructor;
	 *
	 * */

	// set default test data directory
	string home = "../";
	string pdir = "handModelling/Release_2014_5_28/Subject1/";
	path = home + pdir;
	filename = "000001_depth.bin";

	// set camera parameters; given by test set
	imgW = 240;
	imgH = 320;
	scale = 1.0;
	to_cm = true; // boolean that decides whether convert all units to cm
	downsample = false;
	focal_len = 241.42;
	camera_calibration << 241.42 << 0.0    << 160.0 << endr
					   << 0.0	 << 241.42 << 120.0 << endr
					   << 0.0	 << 0.0	   << 1.0	<< endr;

}

void observedmodel::init_observation(string dpath, string dfname, bool mm_to_cm,
									 int imW, int imH, double foclen,
									 bool downsampling) {
	/*
	 * initialise the observed model; updating the default parameters
	 *
	 */
	// for the .bin files: imW = 240, imH = 320
	this->path = dpath;
	this->filename = dfname;
	this->imgW = imW;
	this->imgH = imH;
	this->to_cm = mm_to_cm;
	this->downsample = downsampling;
	this->focal_len = foclen;
	this->img_center << imH/2. << imW/2. << endr;
	this->camera_calibration << foclen << 0.0    << imH/2. << endr
							 << 0.0	   << foclen << imW/2. << endr
							 << 0.0	   << 0.0	 << 1.0	   << endr;
	load_data(); // load and store depthmap to this->depthmap
	get_observed();

}

void observedmodel::get_observed() {
	/*
	 * project the depth map onto 3D forming a point cloud
	 *
	 * computes the distance transform of the inversed depth map
	 */
	depth_to_ptncloud(this->pointcld);
	dist_transform(this->disttran);

}

void observedmodel::set_mm_to_cm(bool mm_to_cm) {
	this->to_cm = mm_to_cm;
}

void observedmodel::set_focal_len(double nfocal) {
	this->focal_len = nfocal;
}

void observedmodel::set_img_center(vec ncenter) {
	this->img_center = ncenter;
}

void observedmodel::set_filename(string name) {
	this->filename = name;
}

void observedmodel::set_path(string newdir) {
	this->path = newdir;
}

void observedmodel::show_depthmap() {
	/*
	 * show the current depth map using OpenCV
	 *
	 */
    cv::Mat opencv_mat(this->imgH, this->imgW,
    					CV_64FC1, this->depthmap.memptr());
    cv::Mat cv_depth(opencv_mat.t());
    cv::namedWindow("depthmap", cv::WINDOW_AUTOSIZE ); // Create a window for display.
    cv::imshow("depthmap", cv_depth); // Show our image inside it.
    cv::waitKey(0);
}

void observedmodel::depth_to_ptncloud(mat &ptncloud) {
	/*
	 * output: ptncloud == point cloud from depth
	 *
	 * solving the following camera transformation:
	 *
	 * const*[u,v,1].T = K*[X,Y,Z].T; where:
	 * const = constant lambda --> gives the homogeneous 2D coordinate
	 * [u,v,1] = homogeneous 2D image coordinate
	 * [X,Y,Z] = 3D euclidean coordinate
	 * K = camera calibration matrix of the form:
	 * [[f, 0, cx], [0, f, cy], [0, 0, 1]]
	 * --> f = focal length, cx & cy = image center
	 *
	 * this can be solved since Z == depth is known; the equations used
	 * are apparent once the matrix multiplication is expanded
	 *
	 * */

	// initialise some containers
	vec x = linspace(0, this->imgW-1, this->imgW);
	vec y = linspace(0, this->imgH-1, this->imgH);
	vec u = ones<vec>(this->imgW);
	vec v = ones<vec>(this->imgH);
	mat ugrid, vgrid, wgrid; // container for X and Y

	// now solve for X and Y; 
	// note: if depth(u,v) == 0 ==> X = Y = Z = 0;
	// such points do not need to be considered in the cost function
	ugrid = u*y.t() - this->img_center(0);
	vgrid = x*v.t() - this->img_center(1);

	ugrid = ugrid % this->depthmap / this->focal_len; // % <==> .*
	vgrid = vgrid % this->depthmap / this->focal_len;
	wgrid = this->depthmap;


	// reshape the result and put them in one container
	// of shape (imgH*imgW, 3) for easier access to every point
	int total_elem = this->imgH * this->imgW;
	ugrid = ugrid.t(); // need to transpose because arma::mat is col major
	vgrid = vgrid.t(); // ==> reshape is done through columns reordering
	wgrid = wgrid.t();
	ugrid.reshape(total_elem, 1); // returns null
	vgrid.reshape(total_elem, 1);
	wgrid.reshape(total_elem, 1);

	uvec nonzero_index = find(wgrid); // find non-zero depth
	ptncloud = zeros<mat>(nonzero_index.n_rows, 3); // reshape ptncloud container
	ptncloud.col(0) = ugrid(nonzero_index); // only need the nonzero points
	ptncloud.col(1) = vgrid(nonzero_index) * -1;
	ptncloud.col(2) = wgrid(nonzero_index) * -1;

	/*
	 * compute the pixel scale of the depth map (i.e. how does pixel distance
	 * translates to real world distance.
	 *
	 * this is done by averaging the real world distance of every two adjacent
	 * pixels (i.e. two neighbouring points in ptncloud)
	 *
	 * */
	double rad  = 2.0;
	mat cens    = zeros<mat>(nonzero_index.n_rows, 3);
	cens.col(0) = ugrid(nonzero_index);
	cens.col(1) = vgrid(nonzero_index);
	cens.col(2) = wgrid(nonzero_index);

	mat edgs    = zeros<mat>(nonzero_index.n_rows, 3);
	edgs.col(0) = ugrid(nonzero_index) + rad;
	edgs.col(1) = vgrid(nonzero_index);
	edgs.col(2) = wgrid(nonzero_index);

	mat projection;
	mat cprojected;
	mat eprojected;
	rowvec dist_norm2;
	rowvec cmPerPixel;
	projection = this->camera_calibration * cens.t();
	cprojected = projection.rows(0,1) / repmat(projection.row(2),2,1);
	cprojected = floor(cprojected);

	projection = this->camera_calibration * edgs.t();
	eprojected = projection.rows(0,1) / repmat(projection.row(2),2,1);
	eprojected = floor(eprojected);

	dist_norm2 = sqrt(square(eprojected.row(0) - cprojected.row(0)) +
					  square(eprojected.row(1) - cprojected.row(1)));


	uvec nonzeroId = find(dist_norm2);
	cmPerPixel     = rad / dist_norm2.cols(nonzeroId);

	this->scale = arma::mean(cmPerPixel);

	if (this->downsample) {
		/*
		 * downsample ptncloud
		 *
		 * */

		int num_samples = 250;
		int sample_f = ptncloud.n_rows / num_samples;

		uvec sample_idx = linspace<uvec>(0,num_samples-1,num_samples) * sample_f;

		ptncloud = ptncloud.rows(sample_idx);

	}

}

void observedmodel::downsample_ptncloud(uvec &rows_id) {
	/*
	 * output: rows_id = the index of points which are selected
	 *
	 * alternative point cloud down sampling function
	 * i.e. implement contour only down-sampling
	 */
	mat dmap = this->depthmap;
	uvec nonzeros = find(dmap!=0); // find all nonzero elements
	dmap(nonzeros).fill(1);

	cv::Mat cvdepthmp(dmap.n_cols, dmap.n_rows, CV_64F, dmap.memptr());
	cvdepthmp = cvdepthmp.t();
	cvdepthmp.convertTo(cvdepthmp, CV_8UC1);

	vector<vector<cv::Point> > contours;
	vector<cv::Vec4i> hierarchy;

	/// Find contours
	findContours(cvdepthmp, contours, hierarchy,
				 CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));


	vector<cv::Point> contour_ptns = contours[0];
	uint increment = 3;
	uint vec_size = contour_ptns.size() / increment;

	rows_id = zeros<uvec>(vec_size);

	for (uint i = 0; i < vec_size; i++) {
		rows_id(i) = 320*(contour_ptns[i*increment].y) + contour_ptns[i*increment].x;
	}

	bool PLOT = false;
	if (PLOT) {

		// Draw contours
		cv::Mat drawing = cv::Mat::zeros(cvdepthmp.size(), CV_8UC3 );
		for(uint i = 0; i< contours.size(); i++ ) {
		   cv::Scalar color = cv::Scalar(0,255,0);
		   cv::drawContours( drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point() );
		}

		// Show in a window
		cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
		cv::imshow( "Contours", drawing );
		cv::waitKey(0);
	}

}

void observedmodel::load_data() {
	/*
	 * load the .bin files (i.e. the depth maps)
	 *
	 */
	string full = this->path + this->filename;

	// get the filesize and create a buffer to store the data
	struct stat statbuf;
	stat(full.c_str(), &statbuf);
	int fsize = statbuf.st_size;
	int bsize = fsize / sizeof(float); // data in file is stored as float

    double buffer[bsize]; // type = double --> can be directly placed in a mat
	float in; // temp-var to store the float data == depth-val

	// load .bin file using fstream
    ifstream fin(full.c_str(), ios::in | ios::binary);
    if (!fin.is_open()) {
          cerr << "error: open file for input failed!" << endl;
          abort();
    }

    for (int i=0; i<bsize; i++) {
    	fin.read((char*)&in, sizeof(float));
    	buffer[i] = (double)in;
    }

    fin.close(); // close the file after loading

    mat depth(&buffer[0], this->imgH, this->imgW, false);

    if (this->to_cm) {
    	depth = depth / 10.; // depthmap originally in unit of mm
    }

    this->depthmap = depth.t(); // (240, 320)

}


void observedmodel::invert_depthmap(mat &depthmp, bool display) {
	/*
	 * invert the depth map to perform distance transform; done simply by
	 * finding the zero and nonzero elements and replace them by the opposite;
	 * i.e. nonzeros --> 0; zeros --> 1
	 *
	 * */
	int nrows = depthmp.n_rows, ncols = depthmp.n_cols;
	depthmp.reshape(nrows*ncols, 1);

	uvec zerosidx = find(depthmp==0); // find all zero element index
	uvec nonzeros = find(depthmp!=0); // find all nonzero elements

	depthmp(zerosidx).fill(255.0);
	depthmp(nonzeros).fill(0.0);

	depthmp.reshape(nrows, ncols);

	if (display) {
		cv::Mat cvdepthmp(ncols, nrows, CV_64F, depthmp.memptr());
		cv::namedWindow("depthmap", cv::WINDOW_AUTOSIZE ); // Create a window for display.
		cv::imshow("depthmap", cvdepthmp); // Show our image inside it.
		cv::waitKey(0);
	}

}

void observedmodel::dist_transform(mat &dist_trans) {
	/*
	 *  calculate the distance transform using the inverted depth map
	 *
	 * */
	mat invdepth = this->depthmap;
	invert_depthmap(invdepth, false); // display = false
	cv::Mat cvdepthmp(invdepth.n_cols, invdepth.n_rows, CV_64F, invdepth.memptr());
	cvdepthmp = cvdepthmp.t();
	cvdepthmp.convertTo(cvdepthmp, CV_8UC1);
	cv::Mat cvdist;
	cv::distanceTransform(cvdepthmp, cvdist, CV_DIST_L2, 5);



	fmat dist(cvdist.ptr<float>(), cvdist.cols, cvdist.rows);
	dist = dist.t();

	dist_trans = conv_to<mat>::from(dist);

	bool debug = false;
	if (debug) {
		cv::namedWindow("disttrans", cv::WINDOW_AUTOSIZE ); // Create a window for display.
		cv::imshow("disttrans", cvdist); // Show our image inside it.
		cv::waitKey(0);
		dist_trans.save("dist_trans.dat", raw_ascii);
		show_depthmap();
	}

}


/*
 * getter functions
 * */
mat observedmodel::get_cam_mat() const {
	return (this->camera_calibration);
}

mat observedmodel::get_depth() const {
	return (this->depthmap);
}

mat* observedmodel::get_camera_mat() {
	return (&this->camera_calibration);
}

mat* observedmodel::get_depthmap() {
	return (&this->depthmap);
}

mat* observedmodel::get_disttran() {
	return (&this->disttran);
}

mat* observedmodel::get_ptncloud() {
	return (&this->pointcld);
}

vec observedmodel::get_img_center() const {
	return (this->img_center);
}

double observedmodel::get_focal_len() const {
	return (this->focal_len);
}

double observedmodel::get_img_scale() const {
	return (this->scale);
}

int observedmodel::get_imgH() const {
	return (this->imgH);
}

int observedmodel::get_imgW() const {
	return (this->imgW);
}


observedmodel& observedmodel::next_frame(string next) {
	/*
	 * update the oberserved model
	 *
	 */
	set_filename(next);
	load_data();
	get_observed();
	return (*this);

}
