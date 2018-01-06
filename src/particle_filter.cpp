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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    const int num_particles = 50;
    default_random_engine gen;

    // creates a normal (Gaussian) distribution for x, y and theta.
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1;
        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    for (auto& p : particles) {
        const double p_x = p.x;
        const double p_y = p.y;
        const double p_theta = p.theta;
        const double yr_dt = yaw_rate * delta_t;

        if (fabs(yaw_rate) < 0.001)
        {
            p.x = p.x + velocity * delta_t * cos(p.theta);
            p.y = p.y + velocity * delta_t * sin(p.theta);
        }
        else
        {
            p.x = p_x + velocity / yaw_rate * (sin(p_theta + yr_dt) - sin(p_theta));
            p.y = p_y + velocity / yaw_rate * (cos(p_theta) - cos(p_theta + yr_dt));
            p.theta = p_theta + yr_dt;
        }

        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
    for(auto& pred : predicted) {
        int min_id = -1;
        double min_dist = -1;
        for(auto& ob : observations) {
            double distance = dist(pred.x, pred.y, ob.x, ob.y);
            if (min_id == -1 || min_dist > distance ) {
                min_dist = distance;
                min_id = ob.id;
            }
        }
        pred.id = min_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    double total_weight = 0;
    for (auto& p : particles) {
        const double p_x = p.x;
        const double p_y = p.y;
        const double p_theta = p.theta;

        vector<Map::single_landmark_s> in_range_landmarks;
        in_range_landmarks.clear();
        for (auto& landmark : map_landmarks.landmark_list) {
            if (sensor_range >= dist(p_x, p_y, landmark.x_f, landmark.y_f))
                in_range_landmarks.push_back(landmark);
        }

        for (auto& ob : observations) {
            // transform to map coordinate
            LandmarkObs map_ob;
            map_ob.x = p_x + (cos(p_theta) * ob.x) - (sin(p_theta) * ob.y);
            map_ob.y = p_y + (sin(p_theta) * ob.x) + (cos(p_theta) * ob.y);

            int min_id = -1;
            double min_dist = -1;
            for (auto& landmark : in_range_landmarks) {
                double distance = dist(map_ob.x, map_ob.y, landmark.x_f, landmark.y_f);
                if (min_id == -1 || min_dist > distance ) {
                    min_dist = distance;
                    min_id = landmark.id_i;
                }
            }
            map_ob.id = min_id;

            const double sig_x = std_landmark[0];
            const double sig_y = std_landmark[1];
            const double mu_x = map_landmarks.landmark_list[map_ob.id-1].x_f;
            const double mu_y = map_landmarks.landmark_list[map_ob.id-1].y_f;

            const double gauss_norm = (1/(2 * M_PI * sig_x *  sig_y));

            const double diff_x = (map_ob.x - mu_x);
            const double diff_y = (map_ob.y - mu_y);
            const double exponent = (diff_x * diff_x)/(2 * sig_x * sig_x) + (diff_y * diff_y)/(2 * sig_y * sig_y);
            const double weight = gauss_norm * exp(-exponent);

            p.weight *= weight;
            if (p.weight == 0.0)
            {
                p.weight = std::numeric_limits<double>::epsilon();
            }
        }

        total_weight += p.weight;
    }

    for (int i = 0 ; i < particles.size(); i++) {
        const double w = particles[i].weight;
        const double n_w = particles[i].weight / total_weight;
        particles[i].weight = n_w;
        weights[i] = n_w;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> resampled;
    const int n = particles.size();

    for (int i = 0; i < n; ++i)
    {
        resampled.push_back(particles[d(gen)]);
    }

    particles = resampled;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
