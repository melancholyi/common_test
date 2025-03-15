/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-01-28 03:56:30
 * @FilePath: /mesh_planner/test/cpp/wassersteinDistance.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

double wassersteinDistance(const Eigen::VectorXd& mu1, const Eigen::MatrixXd& Sigma1, 
                           const Eigen::VectorXd& mu2, const Eigen::MatrixXd& Sigma2) {
    // L2 norm of the difference in means
    Eigen::VectorXd deltaMu = mu1 - mu2;
    double normDeltaMu = deltaMu.squaredNorm();

    // Compute the geometric mean of the covariance matrices
    Eigen::MatrixXd sqrtSigma1 = Sigma1.llt().matrixL();
    Eigen::MatrixXd sqrtSigma2 = Sigma2.llt().matrixL();
    Eigen::MatrixXd product = sqrtSigma1 * sqrtSigma2 * sqrtSigma1;
    Eigen::MatrixXd geoMean = product.llt().matrixL().transpose();

    // Trace of the difference in covariance matrices
    Eigen::MatrixXd sumSigma = Sigma1 + Sigma2;
    double traceDiff = sumSigma.trace() - 2 * geoMean.trace();

    // Wasserstein distance
    double distance = std::sqrt(normDeltaMu + traceDiff);
    return distance;
}

int main() {
    // Define the mean vectors and covariance matrices
    Eigen::VectorXd mu1(3);
    mu1 << 1, 2, 3;
    Eigen::MatrixXd Sigma1(3, 3);
    Sigma1 << 1, 0.8, 0,
              0.8, 1, 0,
              0, 0, 1;

    Eigen::VectorXd mu2(3);
    mu2 << 2, 3, 4;
    Eigen::MatrixXd Sigma2(3, 3);
    Sigma2 << 2, 1.6, 0,
              1.6, 2, 0,
              0, 0, 2;

    // Compute the Wasserstein distance
    double distance = wassersteinDistance(mu1, Sigma1, mu2, Sigma2);

    // Output the result
    std::cout << "Wasserstein distance: " << distance << std::endl;

    return 0;
}