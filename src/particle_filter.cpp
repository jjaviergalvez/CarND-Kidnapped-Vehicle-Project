/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Set the number of particles
	num_particles = 50; 

	// random engine
	std::default_random_engine gen; 

	// Create normal (Gaussian) distributions for x, y and psi.
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	Particle p;
	for (int i = 0; i < num_particles; ++i) {

		// creat a random particle
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;

		// add the particle to the vector of particles
		particles.push_back(p);
	}

	// turn on the init flag
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// random engine
	std::default_random_engine gen;

	// Create normal (Gaussian) distributions for x, y and psi.
	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_theta(0, std_pos[2]);
	
	if (yaw_rate==0){
		double v_dt = velocity*delta_t;
		for(int i=0; i<num_particles; i++){
			particles[i].x += v_dt * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += v_dt * sin(particles[i].theta) + dist_y(gen);
			particles[i].theta += dist_theta(gen);
		}
	}
	else{
		double div = velocity/yaw_rate;
		double yawR_dt = yaw_rate * delta_t;
		for(int i=0; i<num_particles; i++){
			particles[i].x += div * ( sin(particles[i].theta + yawR_dt) - sin(particles[i].theta) ) + dist_x(gen);
			particles[i].y += div * ( cos(particles[i].theta) - cos(particles[i].theta + yawR_dt) ) + dist_y(gen);
			particles[i].theta += yawR_dt + dist_theta(gen);
		}
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html


	double prob;
	double d, min_d, tx, ty;
	double sigma_x, sigma_y, mean_x, mean_y, x, y;
	sigma_x = std_landmark[0];
	sigma_y = std_landmark[1];
	double test;

	for(int i=0; i<num_particles; i++){
		prob = 1; //init the prob with one because we will mult
		for(int j=0; j<observations.size(); j++){
			// Transform each observation from local car coordinate system to the map's coordinate system
			// Equation from http://planning.cs.uiuc.edu/node99.html
			x = particles[i].x + observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta);
			y = particles[i].y + observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta);

			// Associate each transformed observation with a land mark identifier
			min_d = dist(map_landmarks.landmark_list[0].x_f, map_landmarks.landmark_list[0].y_f, x, y);
			mean_x = map_landmarks.landmark_list[0].x_f;
			mean_y = map_landmarks.landmark_list[0].y_f;
			for(int k=0; k<map_landmarks.landmark_list.size(); k++){
				d = dist(map_landmarks.landmark_list[k].x_f, map_landmarks.landmark_list[k].y_f, x, y);
				if(d<min_d){
					min_d = d;
					mean_x = map_landmarks.landmark_list[k].x_f;
					mean_y = map_landmarks.landmark_list[k].y_f;
				}
			}

			// Calculate the particle's final weight
			tx = pow(x-mean_x,2) / (2*pow(sigma_x,2));
			ty = pow(y-mean_y,2) / (2*pow(sigma_y,2));
			prob *= exp(-(tx+ty)) / (2*M_PI*sigma_x*sigma_y);
			
		}

		// Set the weight of the i particle
		particles[i].weight = prob;
		weights.push_back(prob);
	}


}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//Some ideas from here http://stackoverflow.com/questions/31153610/setting-up-a-discrete-distribution-in-c
	std::default_random_engine gen;
	
	// Create the distribution with theweights
  	std::discrete_distribution<> distribution(weights.begin(), weights.end());

  	int ID;
  	std::vector<Particle> new_particles = particles;
  	for(int i=0; i<num_particles; i++){
  		ID = distribution(gen);
  		new_particles[i] = particles[ID];
  		new_particles[i].id = i;
  		new_particles[i].weight = weights[ID];
  	}

  	particles = new_particles;

  	//Removes all elements from the vector (which are destroyed), leaving the container with a size of 0
  	weights.clear();

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
