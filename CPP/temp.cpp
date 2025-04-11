/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-01-31 08:04:16
 * @FilePath: /test/CPP/temp.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

const int INPUT_DIM = 3; // Example input dimension

class ExampleClass {
public:
    ExampleClass(const MatrixXd& trainX) : trainX_(trainX) {}

    void pdDistMatpdXYZ(const MatrixXd& predX, const MatrixXd& distMat, MatrixXd& pdMat) {
        MatrixXd temp_mat(distMat.cols(), distMat.rows() * INPUT_DIM);
        pdMat.resize(distMat.cols(), distMat.rows() * INPUT_DIM);

        for (int i = 0; i < distMat.rows(); i++) {
            for (int j = 0; j < distMat.cols(); j++) {
                // Compute the difference between each row of predX and each row of trainX_
                temp_mat.block(j, i * INPUT_DIM, 1, INPUT_DIM) = predX.row(i) - trainX_.row(j);
            }
            // Copy the transposed distMat into pdMat
            pdMat.block(0, i * INPUT_DIM, distMat.cols(), distMat.rows()) = distMat.transpose();
        }
        std::cout << "temp_mat:\n" << temp_mat << std::endl;
        std::cout << "pdMat:\n" << pdMat << std::endl;
    }

    void pdDistMatpdInput(const MatrixXd &predX, const MatrixXd& distMat, MatrixXd& pdDist){ 
        // pdDist.resize(predX.rows(), trainX_.rows() * INPUT_DIM);
        // for(int input_idx = 0 ; input_idx < INPUT_DIM; input_idx++){//input dim  
        //     Eigen::VectorXd vec_pred = predX.col(input_idx);
        //     std::cout << "vec_pred:\n" << vec_pred << std::endl;
        //     Eigen::VectorXd vec_train = this->trainX_.col(input_idx);
        //     std::cout << "vec_train:\n" << vec_train << std::endl;
        //     Eigen::MatrixXd mat_pred  = vec_pred.replicate(1, vec_train.rows());//dim: [pred, train]
        //     std::cout << "mat_pred:\n" << mat_pred << std::endl;
        //     Eigen::MatrixXd mat_train = vec_train.transpose().replicate(vec_pred.rows(), 1); //dim: [pred, train]
        //     std::cout << "mat_train:\n" << mat_train << std::endl;
        //     std::cout << "(mat_pred.array() - mat_train.array()):\n" << (mat_pred.array() - mat_train.array()) << std::endl;
        //     pdDist.block(0, input_idx * trainX_.rows(), vec_pred.rows(), vec_train.rows()) = (mat_pred.array() - mat_train.array()) / distMat.array();
        //     std::cout << "pdDist.block(0, input_idx * trainX_.rows(), vec_pred.rows(), vec_train.rows()):\n" << pdDist.block(0, input_idx * trainX_.rows(), vec_pred.rows(), vec_train.rows()) << std::endl;
        //     std::cout << "pdDist:\n" << pdDist << std::endl;
        // }
        pdDist.resize(predX.rows(), trainX_.rows() * INPUT_DIM);
        for(int input_idx = 0 ; input_idx < INPUT_DIM; input_idx++){//input dim  
            Eigen::VectorXd vec_pred = predX.col(input_idx);
            Eigen::VectorXd vec_train = this->trainX_.col(input_idx);
            Eigen::MatrixXd mat_pred  = vec_pred.replicate(1, vec_train.rows());//dim: [pred, train]
            Eigen::MatrixXd mat_train = vec_train.transpose().replicate(vec_pred.rows(), 1); //dim: [pred, train]
            pdDist.block(0, input_idx * trainX_.rows(), vec_pred.rows(), vec_train.rows()) = (mat_pred.array() - mat_train.array()) / distMat.array();
        }
        std::cout << "pdDist:\n" << pdDist << std::endl;
    }

private:
    MatrixXd trainX_; // Training data
};

enum class eElevState{
    OBSERVED_E = 0, POTENTIAL_TERRAIN_E = 1, POTENTIAL_OBSTACLE_E = 2, OVERHANG_E = 3, UNOBSERVED_E = 4
};

int main() {
    // // Example training data (5 samples, 3 features)
    // MatrixXd trainX(5, 3);
    // trainX << 1, 2, 3,
    //           4, 5, 6,
    //           7, 8, 9,
    //           10, 11, 12,
    //           13, 14, 15;

    // // Example prediction data (3 samples, 3 features)
    // MatrixXd predX(4, 3);
    // predX << 1.5, 2.5, 3.5,
    //          4.5, 5.5, 6.5,
    //          7.5, 8.5, 9.5,
    //          10.5, 11.5, 12.5;

    // // Example distance matrix (3 rows, 5 columns)
    // MatrixXd distMat = MatrixXd::Ones(4, 5);
    // // distMat << 0.1, 0.2, 0.3, 0.4, 0.5,
    // //            0.6, 0.7, 0.8, 0.9, 1.0,
    // //            1.1, 1.2, 1.3, 1.4, 1.5,
    // //            1.6,1.7,1.8,1.9,2.0;

    // // Initialize the class with training data
    // ExampleClass example(trainX);

    // // Output matrix for partial derivatives
    // MatrixXd pdMat;

    // // Compute the partial derivative matrix
    // example.pdDistMatpdInput(predX, distMat, pdMat);

    // Print the result
    // std::cout << "pdMat:\n" << pdMat << std::endl;
    std::cout << std::flush << std::endl;
    eElevState state;
    state = eElevState::OBSERVED_E;
    uint8_t state_value = static_cast<uint8_t>(state);
    std::cout << "state: " << state_value << std::endl;
    state = eElevState::POTENTIAL_TERRAIN_E;
    state_value = static_cast<uint8_t>(state);
    std::cout << "state: " << state_value << std::endl;
    state = eElevState::POTENTIAL_OBSTACLE_E;
    state_value = static_cast<uint8_t>(state);
    std::cout << "state: " << state_value << std::endl;
    state = eElevState::OVERHANG_E;
    state_value = static_cast<uint8_t>(state);
    std::cout << "state: " << state_value << std::endl;
    state = eElevState::UNOBSERVED_E;
    state_value = static_cast<uint8_t>(state);
    std::cout << "state: " << state_value << std::endl;

    return 0;
}