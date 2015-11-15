#ifndef PSO_H
#define PSO_H

#include "handmodel.h"
#include "observedmodel.h"
#include "costfunc.h"
#include <armadillo>
#include <vector>

using namespace arma;
using namespace std;



class PSO {
private:

	vec theta_min; // lower bound
	vec theta_max; // upper bound
	vec theta_std; // standard deviation for particle initialisation

	double w; // constriction factor
	double c1, c2; // cognitive && social component
	double minstep, minfunc;

	int maxiter;

	void generate_particles(mat &particles, vec &init_postion,
							int num_particles, bool allocation=false);
	void check_constraints(vec &theta, vec &ivelocity);
	void init_velocity(mat &particles, mat &pvelocity, int &num_p);
	void randomise(mat &particles, vec &pcost, vec &x0, int &num_p);
	void restart(mat &particles, mat &velocities, mat &pbest, vec &pcosts, vec &x0);
	void shuffle_particles(mat &particles, mat &velocity, vec &pcost, uvec &permu);

	void NM_simplex(costfunc &optfunc, mat &particles, mat &velocity, vec &pcost);


	void cal_gradient(vec &theta, int &param_sel, costfunc &optfunc,
					  vec &grad, uvec &matchId);
	int wolfe(vec &theta, vec &g_k, uvec &matchId, uvec &sel, double f_k,
					costfunc &optfunc, double &tk, int maxiter=20);
	int goldstein(vec &theta, vec &g_k, uvec &matchId, double f_k,
				  costfunc &optfunc, double &tk, int maxiter=30);
	int armijo(vec &theta, vec &g_k, uvec &matchId, double f_k,
			   costfunc &optfunc, double &tk);

	void cal_grad(vec &x0, uvec &sel, uvec &matchId, costfunc &optfunc, vec &grad);





public:

	PSO();
	void init_PSO(observedmodel *observed, handmodel & hand, int num_particles);
	void update_PSO_gen(int generations, int num_descent=10, int num_clusters=2);

	void refine_init_pose(vec &x0, costfunc &optfunc);
	void dim_restore(vec &theta_in, vec &theta_out);

	int pso_solve(costfunc &optfunc, vec &x0, int num_particles, vec &bestp);
	int pso_evolve(costfunc &optfunc, vec &x0, int num_particles, vec &bestp);
	int pso_optimise(costfunc &optfunc, vec &x0, int num_particles, vec &bestp);
	void set_pso_params(vec &upperbound, vec &lowerbound, vec &std,
						double &omega, double &phip, double &phig, int &maxiter,
						double &minstep, double &minfunc);
	

	
};

#endif
