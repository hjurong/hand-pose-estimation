#ifndef COSTFUNC_H
#define COSTFUNC_H

#include "handmodel.h"
#include "observedmodel.h"
#include <opencv2/core/core.hpp>
#include "opencv2/features2d/features2d.hpp"
#include <armadillo>
#include <vector>

using namespace arma;

class costfunc {

private:
	handmodel *hand;
	observedmodel *observation;

	uvec matchIdx; // unused

	double pairwise_collision(mat &fg1, mat &fg2, vec &rd1, vec &rd2);


public:
	costfunc(handmodel *handM, observedmodel *observed);

	double cal_cost(vec &theta);
	double cal_cost2(vec &theta, uvec &matchId, bool compute_corr, bool debug=false);
	double align_models(vec &spheresR, mat &spheresM, mat &ptncloud, uvec &matchId);
	double depth_penalty(mat &cam_mat, mat &depthmp, mat &spheres, 
						 mat &disttrans, double scale);
	double self_collision_penalty(mat &spheresM, vec &spheresR);
	double bincomp_penalty(mat &spheres);

	double gnd_truth_err(mat &gnd_truth, int frame);

	cv::Mat depthMatch_penalty(mat &spheres);

	void compute_correspondences(mat &ptns, mat &sphM, uvec &matchId);

	uvec get_matchIdx() const;

	
};

#endif 
