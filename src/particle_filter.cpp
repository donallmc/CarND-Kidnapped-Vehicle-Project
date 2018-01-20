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
  if (!is_initialized) {
    num_particles = 100;  //TODO: this should be an init parameter or, at the least, a constant.
    
    normal_distribution<double> norm_dist_x(x, std[0]);
    normal_distribution<double> norm_dist_y(y, std[1]);
    normal_distribution<double> norm_dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
      Particle particle;
      particle.id = i;
      particle.x = norm_dist_x(generator);
      particle.y = norm_dist_y(generator);
      particle.theta = norm_dist_theta(generator);
      particle.weight = 1.0;
      particles.push_back(particle);
    }
    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  for (int i = 0; i < num_particles; i++) {
    if (fabs(yaw_rate) < 0.0001) { // negligible or non-existent
      particles[i].x += velocity * cos(particles[i].theta) * delta_t;
      particles[i].y += velocity * cos(particles[i].theta) * delta_t;
    } else {
      particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      particles[i].theta += yaw_rate * delta_t;
    }

    normal_distribution<double> norm_dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> norm_dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> norm_dist_theta(particles[i].theta, std_pos[2]);

    particles[i].x = norm_dist_x(generator);
    particles[i].y = norm_dist_y(generator);
    particles[i].theta = norm_dist_theta(generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  for (int i = 0; i < observations.size(); i++) {
    int id = -1;
    double min_distance = numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++ ) {
      double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (distance < min_distance) {
	min_distance = distance;
	id = predicted[j].id;
      }
    }
    observations[i].id = id;
  }
}

vector<LandmarkObs> ParticleFilter::findLandmarksInRange(Particle& particle, double sensor_range, const Map &map_landmarks) {
  vector<LandmarkObs> in_range;
  for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
    Map::single_landmark_s landmark = map_landmarks.landmark_list[i];
    if ((fabs((landmark.x_f - particle.x)) <= sensor_range) &&
	(fabs((landmark.y_f - particle.y)) <= sensor_range)) {
      in_range.push_back(LandmarkObs {landmark.id_i, landmark.x_f, landmark.y_f});
    }
  }
  return in_range;
}

vector<LandmarkObs> ParticleFilter::transformToMapCoords(const std::vector<LandmarkObs> &observations, Particle& particle) {
  vector<LandmarkObs> transformed;
  for(int i = 0; i < observations.size(); i++) {
    double x = (cos(particle.theta) * observations[i].x) - (sin(particle.theta) * observations[i].y) + particle.x;
    double y = (sin(particle.theta) * observations[i].x) + (cos(particle.theta) * observations[i].y) + particle.y;
    transformed.push_back(LandmarkObs{ observations[i].id, x, y });
  }
  return transformed;
}

void ParticleFilter::setWeights(Particle& particle, std::vector<LandmarkObs>& observations, std::vector<LandmarkObs>& in_range, double std_landmark[]) {
  particle.weight = 1.0;

  double sigma_x = std_landmark[0];
  double sigma_y = std_landmark[1];
  double sigma_x_2 = pow(std_landmark[0], 2);
  double sigma_y_2 = pow(std_landmark[1], 2);

  for (int i = 0; i < observations.size(); i++) {
    for (int j = 0; j < in_range.size(); j++) {
      if (observations[i].id == in_range[j].id) {
	particle.weight *= (1.0/(2.0 * M_PI * sigma_x * sigma_y)) *
	  exp(-1.0 * ((pow((observations[i].x - in_range[j].x), 2)/(2.0 * sigma_x_2)) + (pow((observations[i].y - in_range[j].y), 2)/(2.0 * sigma_y_2))));
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  for (int i = 0; i < num_particles; i++) {
    vector<LandmarkObs> landmarks_in_range = findLandmarksInRange(particles[i], sensor_range, map_landmarks);
    vector<LandmarkObs> transformed = transformToMapCoords(observations, particles[i]);
    dataAssociation(landmarks_in_range, transformed);
    setWeights(particles[i], transformed, landmarks_in_range, std_landmark);    
  }
}

void ParticleFilter::resample() {
  vector<double> weights; //I'm computing the max here in a single pass rather than retaining a list of weights associated with the object (which is really error-prone!)
  double max_weight = 0;
  for(int i = 0; i < num_particles; i++) {
    if (particles[i].weight > max_weight) {
      max_weight = particles[i].weight;
    }
    weights.push_back(particles[i].weight);      
  }
  
  vector<Particle> resampled;
  uniform_int_distribution<int> particle_lookup(0, num_particles - 1);
  int index = particle_lookup(generator);
  uniform_real_distribution<double> weight_lookup(0.0, max_weight);
  double beta = 0.0;
  for (int i = 0; i < particles.size(); i++) {
    beta += weight_lookup(generator) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled.push_back(particles[index]);
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
