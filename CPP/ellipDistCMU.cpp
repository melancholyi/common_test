/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2024-12-17 04:13:31
 * @FilePath: /mesh_planner/test/cpp/ellipDistCMU.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <chrono>

inline Eigen::MatrixXd invDiag(Eigen::MatrixXd& mat){
    Eigen::MatrixXd temp = mat;
    temp.diagonal().array() = mat.diagonal().array().inverse();
    return temp;
}
inline Eigen::MatrixXd sqrtDiag(Eigen::MatrixXd& mat){
    Eigen::MatrixXd temp = mat;
    temp.diagonal().array() = mat.diagonal().array().sqrt();
    return temp;
}

inline Eigen::MatrixXd invSqrtDiag(Eigen::MatrixXd& mat){
    Eigen::MatrixXd temp = mat;
    temp.diagonal().array() = mat.diagonal().array().inverse().sqrt();
    return temp;
}

inline double getRuntime(const std::chrono::_V2::system_clock::time_point& start){
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
    std::cout << "Time: " << duration.count() << " microseconds" << std::endl;
    return duration.count();
}

inline std::chrono::_V2::system_clock::time_point getTimeNow(){
    return std::chrono::high_resolution_clock::now();
}

// Function to compute the distance between two ellipsoids
Eigen::VectorXd ellipsoidDistance(const Eigen::MatrixXd& b, const Eigen::MatrixXd& B, const Eigen::MatrixXd& c, const Eigen::MatrixXd& C) {  
    //! dim
    int q = B.cols();
    //std::cout << "q:" << q << std::endl;

    /************************** STEP1 ****************************/
    // auto now1 = getTimeNow();
    //! Step 1: Eigen decomposition of B and C. GET \hat{B}, \Lambda_{B} and \hat{C}, \Lambda_{C}
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverB(B);
    Eigen::VectorXd LambdaB = eigensolverB.eigenvalues();
    Eigen::MatrixXd hatB = eigensolverB.eigenvectors();
    //std::cout << "LambdaB:\n" << LambdaB << std::endl;
    //std::cout << "hatB:\n" << hatB << std::endl;
    //std::cout << "Reconstructed B:\n" << hatB * LambdaB.asDiagonal() * hatB.transpose() << std::endl;

    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverC(C);
    Eigen::VectorXd LambdaC = eigensolverC.eigenvalues();
    Eigen::MatrixXd hatC = eigensolverC.eigenvectors(); 
    //std::cout << "LambdaC:\n" << LambdaC << std::endl;
    //std::cout << "hatC:\n" << hatC << std::endl;    
    //std::cout << "Reconstructed C:\n" << hatC * LambdaC.asDiagonal() * hatC.transpose() << std::endl;
    //std::cout << "step1: eigen decompositions" << std::endl;  
    //getRuntime(now1);

    /************************** STEP1 ****************************/
    // //auto now1 = getTimeNow();
    // //! Step 1: Eigen decomposition of B and C. GET \hat{B}, \Lambda_{B} and \hat{C}, \Lambda_{C}
    // Eigen::EigenSolver<Eigen::MatrixXd> eigensolverB(B);
    // Eigen::MatrixXd LambdaB = eigensolverB.pseudoEigenvalueMatrix();
    // Eigen::MatrixXd hatB = eigensolverB.pseudoEigenvectors();
    // //std::cout << "LambdaB:\n" << LambdaB << std::endl;
    // //std::cout << "hatB:\n" << hatB << std::endl;

    
    // Eigen::EigenSolver<Eigen::MatrixXd> eigensolverC(C);
    // Eigen::MatrixXd LambdaC = eigensolverC.pseudoEigenvalueMatrix();
    // Eigen::MatrixXd hatC = eigensolverC.pseudoEigenvectors(); 
    // //std::cout << "LambdaC:\n" << LambdaC << std::endl;
    // //std::cout << "hatC:\n" << hatC << std::endl;
    // //getRuntime(now1);




    /************************** STEP2 ****************************/
    //auto now2 = getTimeNow();
    //! Step 2: Compute B^(1/2), B^(-1/2), and B^(-1), C^(-1)
    Eigen::MatrixXd Bsqrt = hatB * LambdaB.cwiseSqrt().asDiagonal() * hatB.transpose(); // B^(1/2)
    Eigen::MatrixXd Binvsqrt = hatB * LambdaB.cwiseSqrt().cwiseInverse().asDiagonal() * hatB.transpose(); //B^(-1/2)
    Eigen::MatrixXd Binv = hatB * LambdaB.cwiseInverse().asDiagonal() * hatB.transpose(); //B^(-1)
    Eigen::MatrixXd Cinv = hatC * LambdaC.cwiseInverse().asDiagonal() * hatC.transpose(); //C^(-1)
    //std::cout << "Bsqrt:\n" << Bsqrt << std::endl;
    //std::cout << "Binvsqrt:\n" << Binvsqrt << std::endl;
    //std::cout << "Binv:\n" << Binv << std::endl;
    //std::cout << "Cinv:\n" << Cinv << std::endl;
    //getRuntime(now2);

    /************************** STEP2 ****************************/
    // //auto now2 = getTimeNow();
    // //! Step 2: Compute B^(1/2), B^(-1/2), and B^(-1), C^(-1)
    // Eigen::MatrixXd Bsqrt = hatB * LambdaB.cwiseSqrt() * hatB.transpose(); // B^(1/2)
    // Eigen::MatrixXd Binvsqrt = hatB * invSqrtDiag(LambdaB) * hatB.transpose(); //B^(-1/2)
    // Eigen::MatrixXd Binv = hatB * invDiag(LambdaB) * hatB.transpose(); //B^(-1)
    // Eigen::MatrixXd Cinv = hatC * invDiag(LambdaC) * hatC.transpose(); //C^(-1)
    // //std::cout << "Bsqrt:\n" << Bsqrt << std::endl;
    // //std::cout << "Binvsqrt:\n" << Binvsqrt << std::endl;
    // //std::cout << "Binv:\n" << Binv << std::endl;
    // //std::cout << "Cinv:\n" << Cinv << std::endl;
    // //getRuntime(now2);
    

    /************************** STEP3 ****************************/
    //auto now3 = getTimeNow();
    //! Step 3: compute \tilde{C} and \tilde{c}  
    Eigen::MatrixXd tilde_C = Bsqrt * Cinv * Bsqrt;
    //std::cout << "tilde_C:\n" << tilde_C << std::endl;

    Eigen::MatrixXd Binvsqrt_C_Binvsqrt = Binvsqrt * C * Binvsqrt; //A = A^T
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolverBinvsqrt_C_Binvsqrt(Binvsqrt_C_Binvsqrt);  
    Eigen::VectorXd LambdaQsqrt = eigenSolverBinvsqrt_C_Binvsqrt.eigenvalues().cwiseSqrt();
    Eigen::MatrixXd hatQ = eigenSolverBinvsqrt_C_Binvsqrt.eigenvectors();
    Eigen::MatrixXd Binvsqrt_hatQ_LambdaQsqrt_hatQT = Binvsqrt * hatQ * LambdaQsqrt.asDiagonal() * hatQ.transpose();
    //TODO verify. In my experiement, the speed of `colPivHouseholderQr` is more faster than `ColPivHouseholderQR`, 
        //it's against the compariation [official document](https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html) 
    Eigen::VectorXd tilde_c = Binvsqrt_hatQ_LambdaQsqrt_hatQT.fullPivLu().solve(c - b);
    //std::cout << "Binvsqrt_C_Binvsqrt:\n" << Binvsqrt_C_Binvsqrt << std::endl;
    //std::cout << "LambdaQsqrt:\n" << LambdaQsqrt << std::endl;
    //std::cout << "hatQ:\n" << hatQ << std::endl;
    //std::cout << "Binvsqrt_hatQ_LambdaQsqrt_hatQT:\n" << Binvsqrt_hatQ_LambdaQsqrt_hatQT << std::endl;
    //std::cout << "tilde_c:\n" << tilde_c << std::endl;
    //getRuntime(now3);

    
    /************************** STEP4 ****************************/
    //auto now4 = getTimeNow();
    //! Step 4: M1 and \lambda
    Eigen::MatrixXd M1(2*q, 2*q);
    Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q,q);
    M1 << tilde_C, -Iq,
          -1 * tilde_c * tilde_c.transpose(), tilde_C;
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverM1(M1);
    double lambda = eigensolverM1.pseudoEigenvalueMatrix().diagonal().real().minCoeff();   
    //std::cout << "tilde_C:\n" << tilde_C << std::endl;
    //std::cout << "M1:\n" << M1 << std::endl;
    //std::cout << "eigensolverM1.pseudoEigenvalueMatrix():\n" << eigensolverM1.pseudoEigenvalueMatrix() << std::endl;
    //std::cout << "eigensolverM1.pseudoEigenvectors():\n" << eigensolverM1.pseudoEigenvectors() << std::endl;  
    //std::cout << "temp:\n" << eigensolverM1.pseudoEigenvectors() * eigensolverM1.pseudoEigenvalueMatrix() * eigensolverM1.pseudoEigenvectors().inverse() << std::endl;  
    //std::cout << "lambda:\n" << lambda << std::endl;
    //getRuntime(now4);
    

    /************************** STEP5 ****************************/
    //auto now5 = getTimeNow();
    //! Step 5: M2 and compute \alpha, tilde_b
    Eigen::MatrixXd Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt = Binvsqrt * (lambda * Iq    - tilde_C) * Bsqrt;
    Eigen::VectorXd alpha = Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt.fullPivLu().solve(c - b);  
    Eigen::VectorXd tilde_b = -lambda * Binvsqrt * alpha;
    Eigen::MatrixXd M2(2*q, 2*q);
    M2<<Binv, -Iq,
        -tilde_b*tilde_b.transpose(), Binv;
    //std::cout << "Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt:\n" << Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt << std::endl;
    //std::cout << "alpha:\n" << alpha << std::endl;
    //std::cout << "tilde_b:\n" << tilde_b << std::endl;
    //std::cout << "M2:\n" << M2 << std::endl;
    //getRuntime(now5);
    

    /************************** STEP6 ****************************/
    //auto now6 = getTimeNow();
    //! Step 6: mu
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverM2(M2);
    double mu = eigensolverM2.pseudoEigenvalueMatrix().diagonal().real().minCoeff();   
    //std::cout << "eigensolverM2.eigenvalues():\n" << eigensolverM2.eigenvalues() << std::endl;
    //std::cout << "mu:\n" << mu << std::endl;
    //getRuntime(now6);

    /************************** STEP7 ****************************/
    //auto now7 = getTimeNow();
    //! Step 7: compute d* 
    Eigen::MatrixXd mu_Iq_Binv = mu * Iq - Binv;
    Eigen::VectorXd dstar = mu_Iq_Binv.fullPivLu().solve(-mu * lambda * alpha);
    //std::cout << "mu_Iq_Binv:\n" << mu_Iq_Binv << std::endl;
    //std::cout << "dstar:\n" << dstar << std::endl;
    // //getRuntime(now7);


    return dstar;
}

inline void covMat2EllipPmat(const Eigen::MatrixXd& cov, Eigen::MatrixXd& P){
    // Ensure the covariance matrix is square
    assert(cov.rows() == cov.cols());

    // Compute the inverse of the covariance matrix
    Eigen::MatrixXd cov_inv = cov.inverse();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen_slover(cov_inv);
    auto evals = eigen_slover.eigenvalues();
    auto evecs = eigen_slover.eigenvectors();

    // 3-sigma rule
    P = evecs * (evals * 1 / (3*3)).asDiagonal() * evecs.transpose();
}


int main() {
    //======compute ellipsoid distance
    // Eigen::VectorXd p = Eigen::Vector2d(1, 2);
    // Eigen::MatrixXd P; P.resize(2,2);
    // P << 0.625, -0.375,
    //      -0.375, 0.625;   

    // Eigen::VectorXd q = Eigen::Vector2d(6, 2);
    // Eigen::MatrixXd Q; Q.resize(2,2);
    // Q << 0.55555556, -0.44444444 ,
    //      -0.44444444,          0.55555556;

    Eigen::VectorXd p = Eigen::Vector2d(0, 0);
    Eigen::MatrixXd P; P.resize(2,2);
    P << 0.30864198, -0.24691358,
         -0.24691358 , 0.30864198;   

    Eigen::VectorXd q = Eigen::Vector2d(-5, 5);
    Eigen::MatrixXd Q; Q.resize(2,2);
    Q << 0.07407407 ,-0.03703704 ,
         -0.03703704 , 0.07407407;

    Eigen::MatrixXd cov(2,2);
    cov << 1 , 0.8 ,
         0.8 , 1;
    Eigen::MatrixXd Pellip(2,2); Pellip.setZero();
    covMat2EllipPmat(cov, Pellip);
    std::cout << "Pellip:\n" << Pellip << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverB(P);
    Eigen::VectorXd LambdaB = eigensolverB.eigenvalues();
    Eigen::MatrixXd hatB = eigensolverB.eigenvectors();
    std::cout << "LambdaB:\n" << LambdaB << std::endl;
    std::cout << "hatB:\n" << hatB << std::endl;
    std::cout << "Reconstructed B:\n" << hatB * LambdaB.asDiagonal() * hatB.transpose() << std::endl;
    std::cout << "P axis-halflen:\n" << LambdaB.cwiseInverse().cwiseSqrt() << std::endl;

    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverC(Q);
    Eigen::VectorXd LambdaC = eigensolverC.eigenvalues();
    Eigen::MatrixXd hatC = eigensolverC.eigenvectors(); 
    std::cout << "LambdaC:\n" << LambdaC << std::endl;
    std::cout << "hatC:\n" << hatC << std::endl;
    std::cout << "Reconstructed C:\n" << hatC * LambdaC.asDiagonal() * hatC.transpose() << std::endl;
    std::cout << "Q axis-halflen:\n" << LambdaC.cwiseInverse().cwiseSqrt() << std::endl;

    
    


    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    auto dist = ellipsoidDistance(p, P, q, Q);
    
    // 记录开始时间
    auto end = std::chrono::high_resolution_clock::now();

        // 计算运行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "dist: " << dist << std::endl;
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    // Eigen::MatrixXd cov(2,2);
    // cov << 1.0, 0.8,
    //        0.8, 1.0;
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> covSAES(cov);
    // std::cout << "covSAES.eigenvectors():\n" << covSAES.eigenvectors() << std::endl; 
    // std::cout << "covSAES.eigenvalues():\n" << covSAES.eigenvalues() << std::endl;

    

    return 0;
}