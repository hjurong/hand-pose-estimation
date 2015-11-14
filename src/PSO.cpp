#include "PSO.h"
#include "handmodel.h"
#include "observedmodel.h"
#include "costfunc.h"

#include <omp.h>
#include <vector>
#include <string>
#include <armadillo>
#include <opencv2/core/core.hpp>

using namespace arma;
using namespace std;


PSO::PSO() {
	/*
	 * this class implements three different optimisation methods
	 * 1. gradient descent (with numerical differentiation)
	 * 		--> with 3 different line search methods
	 * 2. PSO
	 * 3. gradient descent + PSO
	 *
	 * this class also incoporate different modifications tested on 
	 * these methods; for example: particle restart/re-randomisation
	 */
	this->w = 0.7298;
	this->c1 = 1.49618;
	this->c2 = 1.49618;
	this->maxiter = 100;
	this->minstep = 1e-6;
	this->minfunc = 1e-6;

}

void PSO::set_pso_params(vec &upperbound, vec &lowerbound, vec &std,
						 double &omega, double &phip, double &phig, int &maxIter,
						 double &minStep, double &minFunc) {

	this->theta_min = lowerbound;
	this->theta_max = upperbound;
	this->theta_std = std;
	this->w = omega;
	this->c1 = phip;
	this->c2 = phig;
	this->maxiter = maxIter;
	this->minstep = minStep;
	this->minfunc = minFunc;

}

void PSO::generate_particles(mat &particles, vec &x0, int num_p, bool allocation) {
	// load the parameters of the base hand;
	// then apply random perturbation to the model

	int param_dim = x0.n_rows; // dimension of particle
	particles = randn<mat>(param_dim, num_p); // every COLUMN is a theta
//	particles = 2*randu<mat>(param_dim, num_p) - 1; // every COLUMN is a theta
	mat init_pos = repmat(x0, 1, num_p);
	mat init_std = repmat(this->theta_std, 1, num_p);

	particles = init_pos + particles % init_std; // % ==> element-wise multiplication


}

void PSO::restart(mat &particles, mat &velocities,
					mat &pbest, vec &pcosts, vec &x0) {

	int num_p = particles.n_cols;
	int p_dim = particles.n_rows;

	uvec sorted_cost_idx = sort_index(pcosts); // ascending order; i.e. best -> worst
	uvec reinit_part_idx = sorted_cost_idx.rows(num_p/3, num_p-1);

	mat new_part_pos = repmat(x0, 1, reinit_part_idx.n_rows);
	mat new_hperturb = (randn(p_dim, reinit_part_idx.n_rows)) %
						repmat(this->theta_std, 1, reinit_part_idx.n_rows);


	new_part_pos = new_part_pos + new_hperturb;

	velocities.cols(reinit_part_idx).fill(0.0);
	particles.cols(reinit_part_idx) = new_part_pos;
	pcosts.rows(reinit_part_idx).fill(1e20);
	pbest.cols(reinit_part_idx) = new_part_pos;


}

void PSO::randomise(mat &particles, vec &pcost, vec &x0, int &num_p) {

	int param_dim = particles.n_rows;

	// get the index of sorted cost in ascending order
	uvec sorted_cost_idx = sort_index(pcost);
	uvec psel = randi<uvec>(2, distr_param(0, num_p/2-1));

//	pcost.print("cost");
//	sorted_cost_idx.print("sorted idx");

	int bestc = sorted_cost_idx(0); // psel.at(0); //
	int start = num_p / 2;
//	int ninit = start + num_p / 4;
//	int npert = ninit + 3*num_p / 20;
	int nslic = num_p - 1;


	uvec reinit_idx = sorted_cost_idx.rows(start,nslic);
	mat submat = randn(param_dim, reinit_idx.n_rows);
	mat init_pos = repmat(particles.col(bestc), 1, reinit_idx.n_rows);
//	mat init_pos = repmat(x0, 1, reinit_idx.n_rows);
	mat init_std = repmat(this->theta_std, 1, reinit_idx.n_rows);

//	particles.print("before");

	submat = init_pos + submat % (init_std);
	particles.cols(reinit_idx) = submat;



//	reinit_idx = sorted_cost_idx.rows(ninit+1,npert);
//	reinit_idx = sorted_cost_idx.rows(start, ninit);
//	submat = randn(reinit_idx.n_rows, param_dim);
//	init_std = repmat(this->theta_std.t(), reinit_idx.n_rows, 1);
//	particles.rows(reinit_idx) += submat % (init_std);

//	particles.print("after");

//	reinit_idx = sorted_cost_idx.rows(npert+1, nslic);
////	rowvec bestp = particles.row(bestc);
////	uvec gpt = linspace<uvec>(0,5,6);
////	particles.submat(reinit_idx, gpt) = repmat(bestp.subvec(span(0,5)),
////												reinit_idx.n_rows, 1);
//	rowvec bestp = particles.row(psel.at(1));
//	uvec gpt = linspace<uvec>(6,25,20);
//	particles.submat(reinit_idx, gpt) = repmat(bestp.subvec(span(6,25)),
//												reinit_idx.n_rows, 1);

}

void PSO::init_velocity(mat &particles, mat &pvelocity, int &num_p) {


	mat min_grid = repmat(this->theta_min.t(), num_p, 1);
	mat max_grid = repmat(this->theta_max.t(), num_p, 1);

	pvelocity = randu(num_p, particles.n_cols);

	min_grid -= particles;
	max_grid -= particles;

	pvelocity = min_grid + pvelocity % (max_grid - min_grid);

}


void PSO::shuffle_particles(mat &particles, mat &velocity,
							vec &pcost, uvec &permu) {

	permu = shuffle(permu);
	particles = particles.rows(permu);
	velocity = velocity.rows(permu);
	pcost = pcost.rows(permu);

}


void PSO::dim_restore(vec &theta_in, vec &theta_out) {
	/*
	 * theta_in.shape = (22,1)
	 * theta_out.shape = (26,1)
	 *
	 * apply the constraint: DIP = 2/3 PIP
	 *
	 */

	theta_out.rows(0,5) = theta_in.rows(0,5); // g_rot and g_pos
	theta_out.rows(6,9) = theta_in.rows(6,9); // thumb angles
	theta_out.rows(10,12) = theta_in.rows(10,12); // index angles EXCEPT DIP
	theta_out(13) = 2./3 * theta_in(12); // index PIP
	theta_out.rows(14,16) = theta_in.rows(13,15); // middle angles EXCEPT DIP
	theta_out(17) = 2./3 * theta_in(15); // middle PIP
	theta_out.rows(18,20) = theta_in.rows(16,18); // ring angles EXCEPT DIP
	theta_out(21) = 2./3 * theta_in(18); // ring PIP
	theta_out.rows(22,24) = theta_in.rows(19,21); // little angles EXCEPT DIP
	theta_out(25) = 2./3 * theta_in(21);

}


void PSO::cal_grad(vec &theta, uvec &sel, uvec &matchId, costfunc &optfunc,
				   vec &grad) {

	grad = zeros<vec>(theta.n_rows);

	int len = sel.n_rows;

	for (int i=0; i<len; i++) {
		if (sel(i)) {
			double eps = 1e-5;

			// param_sel, of form: [0 0 .. 1 .. 0] (i.e. 1 specifies the parameter),
			// selects the parameter on which gradient descent is performed;
			vec xph = theta;
			vec xmh = theta;

			xph.row(i) += eps;
			xmh.row(i) -= eps;

			// estimate gradient using:
			// [f(x+h) - f(x-h)] / [2h]
			double f_xph = optfunc.cal_cost2(xph, matchId, false);
			double f_xmh = optfunc.cal_cost2(xmh, matchId, false);
			double grad_val = (f_xph - f_xmh) / (2*eps);

			grad.row(i) = grad_val;
		}
	}
}

void PSO::refine_init_pose(vec &x0, costfunc &optfunc) {

	int len = 2;
	int start_idx[7] = {0, 3, 6, 10, 14, 18, 22};
	int end_idx[7] = {2, 5, 9, 13, 17, 21, 25};

	for (int i=0; i<len; i++) {

		double eps = 1e-6;
		double tol = 1;
		int cnt = 0;
		int iter = 0;
		int maxiter = 15;

		while (tol > eps && iter < maxiter && cnt < 1) {

			uvec matchId;
			uvec param_sel = zeros<uvec>(x0.n_rows);

			param_sel.rows(span(start_idx[i],end_idx[i])).fill(1);

			vec grad;

			double f_k = optfunc.cal_cost2(x0, matchId, true);
			cal_grad(x0, param_sel, matchId, optfunc, grad);

			double tk = 0;
			goldstein(x0, grad, matchId, f_k, optfunc, tk, 30);

			if (tk == 0) {
				cnt += 1;
			}

			x0 = x0 - tk*grad;

			vec temp = sqrt(sum(grad % grad, 0));
			tol = temp(0);
//			cout << c << "--" << tk << "--" << tol << endl; // debug
			iter += 1;

		}
	}


}


void PSO::NM_simplex(costfunc &optfunc, mat &particles, mat &velocity, vec &pcost) {

	cout << pcost << "--in--" << endl;

	double alpha = 1., gamma = 2., p = -0.5, sigma = 0.5;
	int num_p = particles.n_cols;
	int n_dim = particles.n_rows;

	uvec indexes = sort_index(pcost);

	uvec idx_subset = indexes.rows(0, n_dim-1);

	mat par_subset = particles.cols(idx_subset);
	vec cost_subset = pcost.rows(idx_subset);

	int worst_idx = n_dim-1;
	int best_idx = 0;

	int maxiter = 100;

	for (int k=0; k<maxiter; k++) {

		vec x_centroid = (sum(par_subset, 1) - par_subset.col(worst_idx)) / (num_p - 1.);

		vec x_reflection = x_centroid + alpha * (x_centroid - par_subset.col(worst_idx));

		double f_reflection = optfunc.cal_cost(x_reflection);

		if (cost_subset(best_idx) <= f_reflection && f_reflection < cost_subset(worst_idx-1)) {

			par_subset.col(worst_idx) = x_reflection;
			cost_subset(worst_idx) = f_reflection;
		}

		else {
			if (f_reflection < cost_subset(best_idx)) {
				vec x_expansion = x_centroid + gamma * (x_centroid - par_subset.col(worst_idx));

				double f_expansion = optfunc.cal_cost(x_expansion);

				if (f_expansion < f_reflection) {
					par_subset.col(worst_idx) = x_expansion;
					cost_subset(worst_idx) = f_expansion;
				}

				else {
					par_subset.col(worst_idx) = x_reflection;
					cost_subset(worst_idx) = f_reflection;
				}
			}

			else {
				vec x_contraction = x_centroid + p * (x_centroid - par_subset.col(worst_idx));

				double f_contraction = optfunc.cal_cost(x_contraction);

				if (f_contraction < pcost(worst_idx)) {
					par_subset.col(worst_idx) = x_contraction;
					cost_subset(worst_idx) = f_contraction;
				}

				else {
					for (int i=1; i<n_dim; i++) {
						par_subset.col(i) = par_subset.col(0) + sigma * (par_subset.col(i)-par_subset.col(0));
					}
				}
			}
		}

		uvec idx = sort_index(cost_subset);
		cost_subset = cost_subset.rows(idx);
		par_subset = par_subset.cols(idx);

	}

	particles.cols(idx_subset) = par_subset;
	velocity.cols(idx_subset).fill(0.0);
	pcost.rows(idx_subset) = cost_subset;

	cout << pcost << endl;


}


void PSO::check_constraints(vec &theta, vec &ivelocity) {
    
    uvec mark1 = find(theta < this->theta_min);
    uvec mark2 = find(theta > this->theta_max);

    theta(mark1) = this->theta_min(mark1);
    theta(mark2) = this->theta_min(mark2);

    ivelocity(mark1).fill(0.);
    ivelocity(mark2).fill(0.);

//    ivelocity(mark1) *= -0.5;
//    ivelocity(mark2) *= -0.5;

}


void PSO::cal_gradient(vec &theta, int &param_sel, costfunc &optfunc,
					   vec &mgradient, uvec &matchId) {

	double eps = 1e-5;

	// param_sel, of form: [0 0 .. 1 .. 0] (i.e. 1 specifies the parameter),
	// selects the parameter on which gradient descent is performed;
	vec xph = theta;
	vec xmh = theta;

	xph.row(param_sel) += eps;
	xmh.row(param_sel) -= eps;

	// estimate gradient using:
	// [f(x+h) - f(x-h)] / [2h]
	double f_xph = optfunc.cal_cost2(xph, matchId, false);
	double f_xmh = optfunc.cal_cost2(xmh, matchId, false);
	double grad = (f_xph - f_xmh) / (2*eps);

	mgradient = zeros<vec>(theta.n_rows);
	mgradient.row(param_sel) = grad;

}

int PSO::armijo(vec &theta, vec &g_k, uvec &matchId, double f_k,
		   	    costfunc &optfunc, double &tk) {

	double alpha = 1;
	double c0 = 0.25;
	double tau = 0.5;

	vec p_k = -1*g_k;
	vec theta1 = theta + alpha*p_k;

	double armijo = f_k + c0*alpha*dot(g_k,p_k);
	double f_k1 = optfunc.cal_cost2(theta1, matchId, false);

	int iter = 0;
	while (f_k1 > armijo && iter < 30) {
		iter += 1;

		alpha *= tau;
		theta1 = theta + alpha*p_k;
		f_k1 = optfunc.cal_cost2(theta1, matchId, false);
		armijo = f_k + c0*alpha*dot(g_k,p_k);
	}

	tk = alpha;
	return (iter);
}

int PSO::goldstein(vec &theta, vec &g_k, uvec &matchId, double f_k,
			  	   costfunc &optfunc, double &tk, int maxiter) {

	double a = 0;
	double b = 1e100;
	double alpha = 0.5;
	double t = 2;
	double c = 0.25;

	vec p_k = -1*g_k;

	int iter = 0;
	while (iter < maxiter) {

		iter += 1;

		vec theta1 = theta + alpha*p_k;
		double f_k1 = optfunc.cal_cost2(theta1, matchId, false);
		double armijo = f_k + c*alpha*dot(g_k,p_k);
		double goldstein = f_k + (1-c)*alpha*dot(g_k,p_k);

		if (f_k1 <= armijo) {
			if (f_k1 >= goldstein) {
				tk = alpha;
				return (iter);
			}
			else {
				a = alpha;
				alpha = min(t*alpha, 0.5*(alpha+b));

			}

		}
		else {
			b = alpha;
			alpha = 0.5*(a+alpha);
		}

	}

	tk = 0;
	return (iter);
}

int PSO::wolfe(vec &theta, vec &g_k, uvec &matchId, uvec &param_sel, double f_k,
			   costfunc &optfunc, double &tk, int maxiter) {
	/*
	 * find step length using strong wolfe condition
	 * */

	vec p_k = -1*g_k;

	// parameters
	double alpha = 1;
	double a = 0;
	double b = 1e100;
	double t = 2.0;
	double c0 = 0.25;
	double c1 = 0.75;
	double curvature_c = -c1*dot(g_k, p_k);

	for (int i = 0; i < maxiter; ++i) {
		/* code */

		vec x_kPlus1 = theta - alpha*g_k;

		double f_kPlus1 = optfunc.cal_cost2(x_kPlus1, matchId, false); // double
		double armijo_c = f_k + c0*alpha*dot(g_k, p_k);

		if (f_kPlus1 <= armijo_c) {
			/* code */
			vec grad;
			cal_grad(x_kPlus1, param_sel, matchId, optfunc, grad);

			double g_kPlus1 = abs(dot(p_k, grad));

			if (g_kPlus1 <= curvature_c) {
				/* code */
				tk = alpha;
				return (i);
			}
			else {
				a = alpha;
				alpha = min(t*alpha, (b+alpha)/2.);
			}


		}

		else {
			b = alpha;
			alpha = (alpha+a)/2.;
		}

	}

//	tk = alpha;
//	cout << alpha << endl;
	return (maxiter);

}


int PSO::pso_optimise(costfunc &optfunc, vec &x0, int num_p, vec &bestp) {
	/*
	this is not the main optimisation method; merely used for testing
	gradient descent + pso
	*/
	int param_dim = x0.n_rows;
	mat particles = zeros<mat>(param_dim, num_p);; // array of thetas
	mat pvelocity = zeros<mat>(param_dim, num_p); // particle velocity
	mat pbest_pos = zeros<mat>(param_dim, num_p); // best particle position

//	vec rerand(param_dim); // for per generation re-randomisation
//	vec bounds(4);
//	bounds << 2.5 << 10 << 10 << 10 << endr;
//	rerand.rows(0,5).fill(0.0);
//	rerand.rows(6,9) = bounds;
//	rerand.rows(10,13) = bounds;
//	rerand.rows(14,17) = bounds;
//	rerand.rows(18,21) = bounds;
//	rerand.rows(22,25) = bounds;

	vec gbest_pos = zeros<vec>(param_dim); // global best
	vec pcost = zeros<vec>(num_p); // particle costs


	double gbest_cost = 1e100;

	generate_particles(particles, x0, num_p, false);

	#pragma omp parallel for
	for (int i = 0; i < num_p; ++i) {
		/* code */
		vec ctheta = particles.col(i);
		pcost(i) = optfunc.cal_cost(ctheta);
		pbest_pos.col(i) = particles.col(i);

		if (pcost(i) < gbest_cost) {
			/* code */
			gbest_cost = pcost(i);
			gbest_pos = particles.col(i);

		}

	}


	int iter = 1;
	int count = 0;
	int graditer = 10;



	mat cost_evol = zeros<mat>(num_p, (this->maxiter-1)*graditer);
	mat cost_ev2 = zeros<mat>(num_p, (this->maxiter-1));

	mat bcost_evo = zeros<mat>(1, (this->maxiter-1));

	while (iter < this->maxiter) {

		iter += 1;

		uword fmin_id;
		double fmin;


		#pragma omp parallel for
		for (int i = 0; i < num_p; ++i) {

			uvec matchId;
			ivec permu = randi<ivec>(graditer, distr_param(0,param_dim-1));

			bool cal_corr = true; // reset cal_correspondences

			vec ctheta = particles.col(i);
			vec cveloc = pvelocity.col(i);

			for (int m = 0; m < graditer; m++) {
				/* code */

				if (m > 0) {
				    cal_corr = false;
				}


				double f_k = optfunc.cal_cost2(ctheta, matchId, cal_corr);

//				cost_evol(i, (iter-2)*graditer+m) = f_k;

				vec g_k;
				cal_gradient(ctheta, permu(m), optfunc, g_k, matchId);

				double tk = 0;

				// goldstein line search
				goldstein(ctheta, g_k, matchId, f_k, optfunc, tk);

//				armijo(ctheta, g_k, matchId, f_k, optfunc, tk);
//				cout << c <<"--"<< tk << "--" << f_k << endl;

				ctheta = ctheta - tk * g_k;


				f_k = optfunc.cal_cost2(ctheta, matchId, cal_corr);

//				cost_evol(i, (iter-2)*graditer+m) = f_k;

				if (f_k < pcost(i)) {

					pcost(i) = f_k;
					pbest_pos.col(i) = ctheta;
					pvelocity.col(i).fill(0.);

				}


			}

			check_constraints(ctheta, cveloc);
			particles.col(i) = ctheta;
			pvelocity.col(i) = cveloc;


		}


		fmin = pcost.min(fmin_id);

		if (fmin < gbest_cost) {
			gbest_pos = particles.col(fmin_id);
			gbest_cost = fmin;
			count = 0;
		}


//		vec rsu = 2*randu<vec>(num_p)-1;
//		vec sel = randu<vec>(num_p);
//		ivec id = randi<ivec>(num_p, distr_param(6,25));
		mat rp = randu(param_dim, num_p);
		mat rg = randu(param_dim, num_p);
//
//		#pragma omp parallel for
		for (int i=0; i<num_p; i++) {

//			if (count > 0 && sel.at(i) > 0.5) {
//				int index = id.at(i);
//				vec addrand = zeros<vec>(param_dim);
//				addrand.at(index) = rsu.at(i) * rerand.at(index);
//				particles.col(i) += addrand;
//			}
//
////
			pvelocity.col(i) = this->w*pvelocity.col(i) +
							   this->c1*rp.col(i)%(pbest_pos.col(i)-particles.col(i)) +
							   this->c2*rg.col(i)%(gbest_pos - particles.col(i));

			// with constriction factor
//			double thi = 0.7298437881283576;
//			pvelocity.col(i) = thi * (pvelocity.col(i) +
//							   2.8*rp.col(i)%(pbest_pos.col(i)-particles.col(i)) +
//							   1.3*rg.col(i)%(gbest_pos - particles.col(i)));

			particles.col(i) = particles.col(i) + pvelocity.col(i);


			vec ctheta = particles.col(i); // convert to COLUMN vector
			vec cveloc = pvelocity.col(i);
			check_constraints(ctheta, cveloc);
			particles.col(i) = ctheta; // put back to particles matrix
			pvelocity.col(i) = cveloc;


		}

		#pragma omp parallel for
		for (int i=0; i<num_p; i++) {

			vec ctheta = particles.col(i);
			double fx = optfunc.cal_cost(ctheta);
			if (fx < pcost(i)) {
				pcost(i) = fx;
				pbest_pos.col(i) = particles.col(i);

			}

			cost_ev2(i, iter-2) = fx;
		}


		fmin = pcost.min(fmin_id);
		bcost_evo(0, iter-2) = fmin;

		if (fmin < gbest_cost) {
			gbest_pos = particles.col(fmin_id);
			gbest_cost = fmin;
			count = 0;
		}
		else {
			count += 1;
		}



	}

//	cost_evol.save("icppso_wolfe_32p_20i_20s.dat", raw_ascii);
//	cost_ev2.save("icppso_wolfe_32p_20g.dat", raw_ascii);
//	mat d;
//	d.load("frame0004_cost_cmp.dat", raw_ascii);
//	int rng = d.n_cols;
//	rowvec mf = min(cost_evol, 0);
//	rowvec fv = zeros<rowvec>(rng);
//	double fmn= 1e100;
//	for (int j=0; j<rng; j++) {
//		double fval = mf(j);
//		if (fval < fmn) {
//			fmn = fval;
//		}
//
//		fv(j) = fmn;
//
//	}
//	d.insert_rows(0, fv);
//	rowvec psom = zeros<rowvec>(rng);
//	for (int k=0; k<this->maxiter-1; k++) {
//		int start = k*graditer;
//		int end;
//		if (k==this->maxiter-2) {
//			end = (k+1)*graditer-2;
//		}
//		else {
//			end = (k+1)*graditer-1;
//		}
//		psom(span(start, end)).fill(bcost_evo(k));
//	}
//	d.insert_rows(0, bcost_evo);
//	d.save("frame0004_cost_cmp.dat", raw_ascii);

	bestp = gbest_pos;
//	cout << gbest_cost << "--best cost" << endl;

	return (1);


}




int PSO::pso_evolve(costfunc &optfunc, vec &x0, int num_p, vec &bestp) {

	/*
	main optimisation method called in tests
	*/
	arma_rng::set_seed(1000);

	int param_dim = x0.n_rows; // every COLUMN is a particle
	mat particles = zeros<mat>(param_dim, num_p); // array of thetas
	mat pvelocity = zeros<mat>(param_dim, num_p); // particle velocity
	mat pbest_pos = zeros<mat>(param_dim, num_p); // best particle position

	vec rerand(param_dim); // for per generation re-randomisation
	vec bounds(4);
	bounds << 2.5 << 10 << 10 << 10 << endr;
	rerand.rows(0,5).fill(0.0);
	rerand.rows(6,9) = bounds;
	rerand.rows(10,13) = bounds;
	rerand.rows(14,17) = bounds;
	rerand.rows(18,21) = bounds;
	rerand.rows(22,25) = bounds;

	vec gbest_pos = zeros<vec>(param_dim); // global best
	vec pcost = zeros<vec>(num_p); // particle costs


	double gbest_cost = 1e100;

	generate_particles(particles, x0, num_p, false);
//	init_velocity(particles, pvelocity, num_p);


	#pragma omp parallel for
	for (int i = 0; i < num_p; ++i) {
		/* code */
		vec ctheta = particles.col(i);
		pcost(i) = optfunc.cal_cost(ctheta);
		pbest_pos.col(i) = particles.col(i);

		if (pcost(i) < gbest_cost) {
			/* code */
			gbest_cost = pcost(i);
			gbest_pos = particles.col(i);

		}


	}


	int iter = 1;

	int count = 100;
	int nK = 3;
	umat L;

	double W1 = 1./(2*log(2));
	double C1 = 0.5 + log(2);
	double C2 = C1;



	rowvec bcost_evo = zeros<rowvec>(this->maxiter-1);

	while (iter < this->maxiter) {
		iter += 1;

		mat rp = randu<mat>(param_dim, num_p);
		mat rg = randu<mat>(param_dim, num_p);

		vec rsu = 2*randu<vec>(num_p)-1;
		ivec id = randi<ivec>(num_p, distr_param(6,25));
		vec sel = randu<vec>(num_p);



		if (count > 0) {
//			cout << "redefining topology -- " << count << endl; // debug
			L = eye<umat>(num_p, num_p);
			vec R = floor(randu(num_p*nK)*(num_p-1) + 0.5);

			for (int s=0; s<num_p; s++) {
				for (int k=0; k<nK; k++) {
					int r = R(s*nK+k);
					L(s,r) = 1;
				}
			}

//			if (iter % 10 == 0) {
//				restart(particles, pvelocity, pbest_pos, pcost, x0);
//	//				cout << "re-init" << endl;
//			}


		}


//		#pragma omp parallel for
		for (int i = 0; i < num_p; i++) {


//			double MIN = 1e100;
//			int informant = 0;
//			for (int s=0; s<num_p; s++) {
//				if (L(s,i) == 1) {
//					if (pcost.at(s) < MIN) {
//						MIN = pcost(s);
//						informant = s;
//					}
//				}
//			}

			uword ind;
			uvec connection = find(L.col(i) == 1);
			pcost.rows(connection).min(ind);
			int informant = connection(ind);

//			if ((count > 4)) {
//				if (pcost(i) > min(pcost) + 0.5*(max(pcost)-min(pcost))) {
//					int index = id(i);
//					vec addrand = zeros<vec>(param_dim);
//					addrand(index) = rsu(i) * rerand(index);
//					particles.col(i) += addrand;
//				}
//			}


			if (informant == i) {
				pvelocity.col(i) = W1*pvelocity.col(i) +
								   C1*rp.col(i)%(pbest_pos.col(i)-particles.col(i));
			}
			else {
				pvelocity.col(i) = W1*pvelocity.col(i) +
								   C1*rp.col(i)%(pbest_pos.col(i)-particles.col(i)) +
								   C2*rg.col(i)%(pbest_pos.col(informant)-particles.col(i));
			}


			particles.col(i) = particles.col(i) + pvelocity.col(i);


			vec ctheta = particles.col(i); // EVERY COLUMN contains a THETA
			vec cveloc = pvelocity.col(i);
			check_constraints(ctheta, cveloc);
			particles.col(i) = ctheta; // put back to particles matrix
			pvelocity.col(i) = cveloc;


		}


		#pragma omp parallel for
		for (int i=0; i<num_p; i++) {

			vec ctheta = particles.col(i);
			double fx = optfunc.cal_cost(ctheta); // data independent


			if (fx < pcost(i)) {

				pcost(i) = fx;
				pbest_pos.col(i) = particles.col(i);

			}


		}


		uword fmin_id;
		double fmin = pcost.min(fmin_id);
		bcost_evo(iter-2) = fmin;


		if (fmin < gbest_cost) {
			gbest_pos = particles.col(fmin_id);
			gbest_cost = fmin;
			count = 0;
		}
		else {
			count += 1;
		}

		bcost_evo(iter-2) = gbest_cost;

	}
//
//
////	cout << gbest_cost << "--best cost" << endl;
//	cost_evol.save("psolbest_32p_200i.dat", raw_ascii);
	// bcost_evo.save("frame0004_cost_rand.dat", raw_ascii);
//	mat d;
//	d.load("frame0004_cost_cmp.dat", raw_ascii);
//	d.insert_rows(0, bcost_evo);
//	d.save("frame0004_cost_cmp.dat", raw_ascii);


	bestp = gbest_pos;


	return (1);

}
