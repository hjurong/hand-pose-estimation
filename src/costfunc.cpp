#include "costfunc.h"
#include "handmodel.h"
#include "observedmodel.h"

#include <vector>
#include <iostream>
#include <armadillo>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"

using namespace arma;
using namespace std;

costfunc::costfunc(handmodel *handM, observedmodel *observed) {
	/*
	 * takes the 48 spheres hand model and the oboverved model as
	 * input can compute the associated cost
	 */
	this->hand = handM;
	this->observation = observed;
	// set_dist_transform(); // set the inverted depthmap
}

double costfunc::cal_cost2(vec &theta, uvec &matchId, bool compute_corr,
						   bool debug) {
	/*
	 * bool calall, double &term1, double &term2, double &term3
	 * calculate the new cost associated with the new theta;
	 *
	 * firstly, need to update this->hand with the new theta (include transform and
	 * build spheres);
	 *
	 * then, call below functions to calculate penalty associated with each term of the
	 * cost function and return the sum
	 *
	 * */


	mat::fixed<48,3> spheresM;
	this->hand->build_hand_model(theta, spheresM);

	if (compute_corr) {
		// must compute point-sphere correspondence first before
		// computing alignment cost
		compute_correspondences(*(this->observation->get_ptncloud()),
								spheresM, matchId);
	}

	double align_cost = align_models(*(this->hand->get_radii()),
									 spheresM,
									 *(this->observation->get_ptncloud()),
									 matchId);

	double depth_cost = depth_penalty(*(this->observation->get_camera_mat()),
									  *(this->observation->get_depthmap()),
									  spheresM,
									  *(this->observation->get_disttran()),
									  this->observation->get_img_scale());

	double colli_cost = self_collision_penalty(spheresM, *(this->hand->get_radii()));

	if (debug) {
		cout << align_cost << endl;
		cout << depth_cost << endl;
		cout << colli_cost << "\n" << endl;
	}


	double total_cost = align_cost + depth_cost + colli_cost;

	return (total_cost);


}


double costfunc::cal_cost(vec &theta) {
	/*
	 * bool calall, double &term1, double &term2, double &term3
	 * calculate the new cost associated with the new theta;
	 *
	 * firstly, need to update this->hand with the new theta (include transform and
	 * build spheres);
	 *
	 * then, call below functions to calculate penalty associated with each term of the
	 * cost function and return the sum
	 *
	 * */

		mat::fixed<48,3> spheresM;
		this->hand->build_hand_model(theta, spheresM);


		// must compute point-sphere correspondence first before
		// computing alignment cost
		uvec matchId;
		compute_correspondences(*(this->observation->get_ptncloud()),
								spheresM, matchId);

		double align_cost = align_models(*(this->hand->get_radii()),
										 spheresM,
										 *(this->observation->get_ptncloud()),
										 matchId);

		double depth_cost = depth_penalty(*(this->observation->get_camera_mat()),
										  *(this->observation->get_depthmap()),
										  spheresM,
										  *(this->observation->get_disttran()),
										  this->observation->get_img_scale());

//		double depth_cost = depthMatch_penalty(spheresM);

//		double colli_cost = self_collision_penalty(spheresM, *(this->hand->get_radii()));

		double total_cost = align_cost + depth_cost; // + colli_cost;


		return (total_cost);

}


double costfunc::self_collision_penalty(mat &spheresM, vec &spheresR) {
	/*
	 * third term of cost function that penalises self collision; the cost is given
	 * by:
	 *
	 * sum_{i, j}(L(s_i, s_j)^2);
	 *
	 * L(s_i, s_j) = max(r_i + r_j - L2norm(c_i - c_j), 0), where:
	 *
	 * s_i, s_j = the i-th and j-th sphere
	 * r_i, r_j = radius of respective spheres
	 * c_i, c_j = centers of respective spheres
	 *
	 * this is only applied to neighbouring spheres
	 *
	 * */

//  // pointer test //////////////////////////////////////////////////
//  cout << hand << " hand addr" << endl;
//  vec newspacing = this->hand->get_spacing();
//  cout << "new spacing: " << newspacing << endl;
//  //////////////////////////////////////////////////////////////////

	mat::fixed<6,3> tbs, ixs, mds, rgs, lts;
	vec::fixed<6> tbr, ixr, mdr, rgr, ltr;

	tbs = spheresM.rows(2,7);
	ixs = spheresM.rows(12,17);
	mds = spheresM.rows(22,27);
	rgs = spheresM.rows(32,37);
	lts = spheresM.rows(42,47);

	tbr = spheresR.rows(2,7);
	ixr = spheresR.rows(12,17);
	mdr = spheresR.rows(22,27);
	rgr = spheresR.rows(32,37);
	ltr = spheresR.rows(42,47);

	mat::fixed<6,3> *s_array[5] = {&tbs, &ixs, &mds, &rgs, &lts};
	vec::fixed<6> *r_array[5] = {&tbr, &ixr, &mdr, &rgr, &ltr};

	double penalty = 0.0;
	// loop 4 times are there are four pairs of digits to compare
	for (int i=0; i<4; i++) {


		mat smat1 = *s_array[i];

		smat1 = repmat(smat1.t(), 6, 1);
		smat1.reshape(3, 36); // shape == (6,3) replicated 6 times
		smat1 = smat1.t();

		mat smat2 = *s_array[i+1];
		smat2 = repmat(smat2, 6, 1);
//
		vec rvec1 = *r_array[i];
		mat tempr = repmat(rvec1.t(), 6, 1);
		tempr.reshape(36, 1); // 36 rows 1 column
		rvec1 = tempr;

		vec rvec2 = *r_array[i+1];
		rvec2 = repmat(rvec2, 6, 1);

		mat diff = pow(smat2 - smat1, 2); // square difference
		vec dist = sqrt(sum(diff, 1)); // column by column sum --> norm

		dist = rvec2 + rvec1 - dist;
		uvec idx = find(dist > 0);

		penalty += sum(square(dist(idx)));

	}

	return (penalty);
}

double costfunc::pairwise_collision(mat &fg1, mat &fg2, vec &rd1, vec &rd2) {
	int nspheres = fg1.n_rows;
	double pairwise_penalty = 0.0;

	for (int i=0; i<nspheres; ++i) {
		mat diff = fg1 - fg2;
		vec sdis = sqrt(square(diff.col(0)) +
						square(diff.col(1)) + square(diff.col(2)));
		sdis += rd1 + rd2;

		uvec nonzero = find(sdis>0);
		pairwise_penalty += sum(square(sdis(nonzero)));

		fg1.swap_rows(0, nspheres-1);
		rd1.swap_rows(0, nspheres-1);

	}

	return (pairwise_penalty);
}

double costfunc::depth_penalty(mat &cam_mat, mat &depthmp, mat &spheres,
							   mat &disttrans, double scale) {
	/*
	 * second term of the cost function; this forces the sphere model to lie
	 * inside the ptncloud; it is given by:
	 *
	 * B(c, D) =
	 * .........(a) max(0, D(j(c))-cz)
	 * .........(b) dis(j(c), D)
	 *
	 * ==> if the projection of spheresM (u,v) has non-zero depth, apply (a) -->
	 * with D(j(c)) == depth at (u,v) and cz = depth at sphere center
	 *
	 * ==> if the projection of spheresM (u,v) has zero depth, apply (b) -->
	 * i.e. distance between (u,v) to the non-zero depth calculated using
	 * distance transform of the inverted depth map
	 *
	 * */


	vec radii = *(this->hand->get_radii());

	spheres.cols(1,2) *= -1;

	mat projection = cam_mat * spheres.t(); // (3x3) * (3xnspheres)
	projection = projection / repmat(projection.row(2), 3, 1); // to homogeneous
	projection.shed_row(2); // remove the last row, which are all 1's;
	projection = projection.t(); // shape = (nspheres, 2)
	projection = floor(projection); // val --> int for indexing


	double depth_penalty = 0.0;
	int count = projection.n_rows;
	for (int i = 0; i < count; ++i) {
		double dx = projection(i,0);
		double dy = projection(i,1);

		/*
		 * firstly check if the projection of the sphere centers lie inside
		 * the depthmap (i.e. inside 320x240 as described by xbounded and ybounded)
		 *
		 * if so, calculate the cost accordingly
		 *
		 * else, add the maximum distance from dist-transform
		 *
		 * */
		bool xbounded = dx >= 0 && dx < depthmp.n_cols; // end = depthmp.n_rows-1
		bool ybounded = dy >= 0 && dy < depthmp.n_rows;

		if (xbounded && ybounded) {

			double d_jc = depthmp(dy, dx);
			if (d_jc != 0.0) {

				double diff = max(0.0, d_jc-spheres(i, 2)); // depth at u,v

				depth_penalty += diff*diff;
			}
			else {

				double ddis = disttrans(dy, dx) * scale + radii(i);
				depth_penalty += ddis*ddis;
			}

		}
		else {
			/*
			 * if the projection is out side the image (i.e. 320 x 240)
			 * */

			double maxdis = disttrans.max() * scale + radii(i);
			depth_penalty += maxdis*maxdis;
		}
	}

	return (depth_penalty);
}

void costfunc::compute_correspondences(mat &ptns, mat &sphM, uvec &matchId) {

	fmat spheresM = conv_to<fmat>::from(sphM);
	fmat ptncloud = conv_to<fmat>::from(ptns);

//  cout << spheresM.n_rows << "-spr-" << spheresM.n_cols << endl;
//  cout << ptncloud.n_rows << "-ptn-" << ptncloud.n_cols << endl;

	cv::Mat cvspheresM(spheresM.n_cols, spheresM.n_rows, CV_32F, spheresM.memptr());
	cv::Mat cvptncloud(ptncloud.n_cols, ptncloud.n_rows, CV_32F, ptncloud.memptr());

	cvspheresM = cvspheresM.t(); // unit = cm
	cvptncloud = cvptncloud.t(); // converted to cm from mm

	cv::BFMatcher matcher(cv::NORM_L2); // initialise a brute-force matcher
	std::vector<cv::DMatch> matches; // initialise a container for the matches


	matcher.match(cvptncloud, cvspheresM, matches); // perform matching

	int cnt = ptncloud.n_rows;
	matchId = zeros<uvec>(cnt);

	for (int i = 0; i < cnt; i++) {
		matchId.at(i) = matches[i].trainIdx;
	}

}


double costfunc::align_models(vec &spheresR, mat &spheresM,
							  mat &ptncloud, uvec &matchId) {
	/*
	 * this computes the first term of the cost function; i.e.:
	 *
	 * const * sum{for ptn in ptncloud} Dis(ptn, nearest_scenter)^2
	 *
	 * where:
	 *
	 * const = num_spheres / num_ptncloud
	 * ptn = current point in ptncloud
	 * nearest_scenter = the nearest sphere center to the current point
	 *
	 * sphradii.shape = (nspheres, 1)
	 * spheresM.shape = (nspheres, 3)
	 * ptncloud.shape = (npoints, 3)
	 *
	 * */

	mat spheresC = spheresM.rows(matchId);
	vec sphradii = spheresR.rows(matchId);

	mat ptndist = pow(ptncloud - spheresC, 2);
	vec nordist = sqrt(sum(ptndist, 1)); // column by column sum
	vec difdist = pow(abs(nordist - sphradii), 2);

	double lambda = (double)spheresM.n_rows/ptncloud.n_rows;
	double dis = sum(difdist) * lambda; // apply scaling |M|/|P|

	return (dis);

}


uvec costfunc::get_matchIdx() const {
	return (this->matchIdx.rows(span(0,10)));
}

double costfunc::bincomp_penalty(mat &spheres) {

	mat K = *(this->observation->get_camera_mat());

	mat cens = spheres.t();
	cens.rows(1,2) *= -1;

	mat projection = K * cens;
	mat cprojected = projection.rows(0,1) / repmat(projection.row(2),2,1);

	vec R = *(this->hand->get_radii());
	R /= this->observation->get_img_scale();

	int imW = this->observation->get_imgW();
	int imH = this->observation->get_imgH();

//	mat invdepth = this->observation->get_depth();
//	cv::Mat out_img(invdepth.n_cols, invdepth.n_rows, CV_64F, invdepth.memptr());
//	out_img = out_img.t();

	cv::Mat out_img = cv::Mat::zeros(imW,imH,CV_64F);

	int nptns = cprojected.n_cols;
	for (int i=0; i<nptns; i++) {

		double dep = cens(2,i);
		int cx = cprojected(0,i);
		int cy = cprojected(1,i);
		int rad = (int) R(i);

		cv::circle(out_img, cv::Point(cx,cy), rad, cv::Scalar(dep),-1);
	}


	mat dep(out_img.ptr<double>(), out_img.cols, out_img.rows);
	dep = dep.t();
	dep.elem( find(dep) ).ones();

	mat dmap = this->observation->get_depth();
	dmap.elem( find(dmap) ).ones();

	double c = sum(sum(abs(dep-dmap)));

	return (c);
}

cv::Mat costfunc::depthMatch_penalty(mat &spheres) {

	mat K = *(this->observation->get_camera_mat());

	mat cens = spheres.t();
	cens.rows(1,2) *= -1;

	mat projection = K * cens;
	mat cprojected = projection.rows(0,1) / repmat(projection.row(2),2,1);

	vec R = *(this->hand->get_radii());
	R /= this->observation->get_img_scale();



	mat invdepth = this->observation->get_depth();
	cv::Mat out_img(invdepth.n_cols, invdepth.n_rows, CV_64F, invdepth.memptr());
	out_img = out_img.t();

//	int imW = this->observation->get_imgW();
//	int imH = this->observation->get_imgH();
//	cv::Mat out_img = cv::Mat::zeros(imW,imH,CV_64F);

	int nptns = cprojected.n_cols;
	for (int i=0; i<nptns; i++) {

		double dep = cens(2,i);
		int cx = cprojected(0,i);
		int cy = cprojected(1,i);
		int rad = (int) R(i);

		cv::circle(out_img, cv::Point(cx,cy), rad, cv::Scalar(dep,dep,dep),2);
	}

//	cv::namedWindow("disttrans", cv::WINDOW_AUTOSIZE ); // Create a window for display.
//	cv::imshow("disttrans", out_img); // Show our image inside it.
//	cv::waitKey(0);

//	mat dep(out_img.ptr<double>(), out_img.cols, out_img.rows);
//	dep = dep.t();

////	dep.save("projected_model.dat", raw_ascii);
//
//	mat depth_diff = dep - (this->observation->get_depth());
//
//	double cost = sum(sum(abs(depth_diff)));


	return (out_img);

}


double costfunc::gnd_truth_err(mat &gnd_truth, int frame) {

	mat gndTr_joints = gnd_truth.row(frame);
	gndTr_joints.reshape(3,21);

	// assume that the joints have already been computed
	// through "build_hand_model(.)"
	mat hand_joints = this->hand->hand_joints * 10.0; // convert back to mm from cm
	hand_joints.cols(1,2) *= -1;

	mat diff = gndTr_joints.t() - hand_joints;

	vec dist = sqrt(square(diff.col(0)) +
					square(diff.col(1)) +
					square(diff.col(2)));

	uvec sjoint_id(6);
	sjoint_id << 0 << 4 << 8 << 12 << 16 << 20  << endr;

//	cout << diff.rows(sjoint_id) << endl;
//	cout << hand_joints.rows(sjoint_id) << endl;
//	cout << gndTr_joints.cols(sjoint_id).t() << endl;
//	cout << dist(sjoint_id) << endl;

	double c = sum(dist(sjoint_id));


	return (c);
}

