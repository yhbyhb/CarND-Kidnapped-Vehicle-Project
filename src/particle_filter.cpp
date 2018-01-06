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

    const int num_particles = 20;
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
    }

    cout << "init()" << endl;
    for (auto& p : particles) {
        cout << p.id << " " << p.x << " "  << p.y << endl;
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

    cout << "prediction()" << endl;
    for (auto& p : particles) {
    
        cout << "before : "<< p.id << " " << p.x << " "  << p.y << endl;
    
        const double p_x = p.x;
        const double p_y = p.y;
        const double p_theta = p.theta;
        const double yr_dt = yaw_rate * delta_t;
        p.x = p_x + velocity / p_theta * (sin(p_theta + yr_dt) - sin(p_theta)) + dist_x(gen);
        p.y = p_y + velocity / p_theta * (cos(p_theta) - cos(p_theta + yr_dt)) + dist_y(gen);
        p.theta = p_theta + yr_dt + dist_theta(gen);
    
        cout << "after : " << p.id << " " << p.x << " "  << p.y << endl;
    }    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    cout << "dataAssociation()" << endl;

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

    cout << "updateWeights()" << endl;

    double total_weight = 0;
    for (auto& p : particles) {
        const double p_x = p.x;
        const double p_y = p.y;
        const double p_theta = p.theta;

        cout << "    p.x : " << p_x << " p.y : " << p_y << " p.theta : " << p_theta << " p.id : " << p.id << endl;

        for (auto& ob : observations) {

            cout << "    ob.x : " << ob.x << " ob.y : " << ob.y << " ob.id : " << ob.id << endl;

            // transform to map coordinate
            LandmarkObs map_ob;
            map_ob.x = p_x + (cos(p_theta) * ob.x) - (sin(p_theta) * ob.y);
            map_ob.y = p_y + (sin(p_theta) * ob.y) + (cos(p_theta) * ob.y);

            int min_id = -1;
            double min_dist = -1;
            for (auto& landmark : map_landmarks.landmark_list) {
                if (sensor_range < dist(p_x, p_y, landmark.x_f, landmark.y_f))
                    continue;

                double distance = dist(map_ob.x, map_ob.y, landmark.x_f, landmark.y_f); 
                if (min_id == -1 || min_dist > distance ) {                
                    min_dist = distance;
                    min_id = ob.id;
                }
            }
            map_ob.id = min_id;

            const double sig_x = std_landmark[0];
            const double sig_y = std_landmark[1];
            const double mu_x = map_landmarks.landmark_list[map_ob.id].x_f;
            const double mu_y = map_landmarks.landmark_list[map_ob.id].y_f;

            const double gauss_norm = (1/(2 * M_PI * sig_x *  sig_y));

            const double diff_x = (map_ob.x - mu_x);
            const double diff_y = (map_ob.y - mu_y);
            const double exponent = (diff_x * diff_x)/(2 * sig_x * sig_x) + (diff_y * diff_y)/(2 * sig_y * sig_y);
            const double weight = gauss_norm * exp(-exponent);

            p.weight *= weight;

            cout << "    map_ob.x : " << map_ob.x << " map_ob.y : " << map_ob.y << " map_ob.id : " << map_ob.id << endl;
            cout << "        sig_x : " << sig_x << " sig_y : " << sig_y << " mu_x : " << mu_x << " mu_y : " << mu_y << " weight : " << weight << endl;
        }

        weights.push_back(p.weight);
        total_weight += p.weight;
    }

    for (auto& w : weights) {
        w /= total_weight;
        cout << "    weight : "<< w << " total_weight : " << total_weight << endl;
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
