<!--
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-01-25 10:48:13
 * @FilePath: /mesh_planner/test/cpp/README.md
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
-->
```cmake  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
```

# Ellipsoid MindIST
All `Eigen::EigenSlover`  
```CPP
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

    auto now1 = getTimeNow();
    //! Step 1: Eigen decomposition of B and C. GET \hat{B}, \Lambda_{B} and \hat{C}, \Lambda_{C}
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverB(B);
    Eigen::MatrixXd LambdaB = eigensolverB.pseudoEigenvalueMatrix();
    Eigen::MatrixXd hatB = eigensolverB.pseudoEigenvectors();
    //std::cout << "LambdaB:\n" << LambdaB << std::endl;
    //std::cout << "hatB:\n" << hatB << std::endl;

    
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverC(C);
    Eigen::MatrixXd LambdaC = eigensolverC.pseudoEigenvalueMatrix();
    Eigen::MatrixXd hatC = eigensolverC.pseudoEigenvectors(); 
    //std::cout << "LambdaC:\n" << LambdaC << std::endl;
    //std::cout << "hatC:\n" << hatC << std::endl;
    getRuntime(now1);

    auto now2 = getTimeNow();
    //! Step 2: Compute B^(1/2), B^(-1/2), and B^(-1), C^(-1)
    Eigen::MatrixXd Bsqrt = hatB * LambdaB.cwiseSqrt() * hatB.transpose(); // B^(1/2)
    Eigen::MatrixXd Binvsqrt = hatB * invSqrtDiag(LambdaB) * hatB.transpose(); //B^(-1/2)
    Eigen::MatrixXd Binv = hatB * invDiag(LambdaB) * hatB.transpose(); //B^(-1)
    Eigen::MatrixXd Cinv = hatC * invDiag(LambdaC) * hatC.transpose(); //C^(-1)
    //std::cout << "Bsqrt:\n" << Bsqrt << std::endl;
    //std::cout << "Binvsqrt:\n" << Binvsqrt << std::endl;
    //std::cout << "Binv:\n" << Binv << std::endl;
    //std::cout << "Cinv:\n" << Cinv << std::endl;
    getRuntime(now2);
    
    auto now3 = getTimeNow();
    //! Step 3: compute \tilde{C} and \tilde{c}  
    Eigen::MatrixXd tilde_C = Bsqrt * Cinv * Bsqrt;
    //std::cout << "tilde_C:\n" << tilde_C << std::endl;

    Eigen::MatrixXd Binvsqrt_C_Binvsqrt = Binvsqrt * C * Binvsqrt;
    Eigen::EigenSolver<Eigen::MatrixXd> eigenSolverBinvsqrt_C_Binvsqrt(Binvsqrt_C_Binvsqrt);  
    Eigen::MatrixXd LambdaQsqrt = eigenSolverBinvsqrt_C_Binvsqrt.pseudoEigenvalueMatrix().cwiseSqrt();
    Eigen::MatrixXd hatQ = eigenSolverBinvsqrt_C_Binvsqrt.pseudoEigenvectors();
    Eigen::MatrixXd Binvsqrt_hatQ_LambdaQsqrt_hatQT = Binvsqrt * hatQ * LambdaQsqrt * hatQ.transpose();
    Eigen::VectorXd tilde_c = Binvsqrt_hatQ_LambdaQsqrt_hatQT.ldlt().solve(c - b);
    //std::cout << "Binvsqrt_C_Binvsqrt:\n" << Binvsqrt_C_Binvsqrt << std::endl;
    //std::cout << "LambdaQsqrt:\n" << LambdaQsqrt << std::endl;
    //std::cout << "hatQ:\n" << hatQ << std::endl;
    //std::cout << "Binvsqrt_hatQ_LambdaQsqrt_hatQT:\n" << Binvsqrt_hatQ_LambdaQsqrt_hatQT << std::endl;
    //std::cout << "tilde_c:\n" << tilde_c << std::endl;
    getRuntime(now3);

    
    auto now4 = getTimeNow();
    //! Step 4: M1 and \lambda
    Eigen::MatrixXd M1(2*q, 2*q);
    Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q,q);
    M1 << tilde_C, -Iq,
          -1 * tilde_c * tilde_c.transpose(), tilde_C;
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverM1(M1);
    double lambda = eigensolverM1.pseudoEigenvalueMatrix().real().minCoeff();   
    //std::cout << "tilde_C:\n" << tilde_C << std::endl;
    //std::cout << "M1:\n" << M1 << std::endl;
    //std::cout << "eigensolverM1.pseudoEigenvalueMatrix():\n" << eigensolverM1.pseudoEigenvalueMatrix() << std::endl;
    //std::cout << "eigensolverM1.pseudoEigenvectors():\n" << eigensolverM1.pseudoEigenvectors() << std::endl;  
    //std::cout << "temp:\n" << eigensolverM1.pseudoEigenvectors() * eigensolverM1.pseudoEigenvalueMatrix() * eigensolverM1.pseudoEigenvectors().transpose() << std::endl;  
    //std::cout << "lambda:\n" << lambda << std::endl;
    getRuntime(now4);
    
    auto now5 = getTimeNow();
    //! Step 5: M2 and compute \alpha, tilde_b
    Eigen::MatrixXd Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt = Binvsqrt * (lambda * Iq - tilde_C) * Bsqrt;
    Eigen::VectorXd alpha = Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt.ldlt().solve(c -b);
    Eigen::VectorXd tilde_b = -lambda * Binvsqrt * alpha;
    Eigen::MatrixXd M2(2*q, 2*q);
    M2<<Binv, -Iq,
        -tilde_b*tilde_b.transpose(), Binv;
    //std::cout << "Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt:\n" << Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt << std::endl;
    //std::cout << "alpha:\n" << alpha << std::endl;
    //std::cout << "tilde_b:\n" << tilde_b << std::endl;
    //std::cout << "M2:\n" << M2 << std::endl;
    getRuntime(now5);
    
    auto now6 = getTimeNow();
    //! Step 6: mu
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverM2(M2);
    double mu = eigensolverM2.pseudoEigenvalueMatrix().real().minCoeff();   
    //std::cout << "mu:\n" << mu << std::endl;
    //std::cout << "eigensolverM2.eigenvalues():\n" << eigensolverM2.eigenvalues() << std::endl;
    getRuntime(now6);

    auto now7 = getTimeNow();
    //! Step 7: compute d* 
    Eigen::MatrixXd mu_Iq_Binv = mu * Iq - Binv;
    Eigen::VectorXd dstar = mu_Iq_Binv.ldlt().solve(-mu * lambda * alpha);
    //std::cout << "mu_Iq_Binv:\n" << mu_Iq_Binv << std::endl;
    //std::cout << "dstar:\n" << dstar << std::endl;
    getRuntime(now7);


    return dstar;
}


int main() {
    //======compute ellipsoid distance
    Eigen::VectorXd p = Eigen::Vector2d(1, 2);
    Eigen::MatrixXd P; P.resize(2,2);
    P << 1, 0,
         0, 1;   

    Eigen::VectorXd q = Eigen::Vector2d(6, 2);
    Eigen::MatrixXd Q; Q.resize(2,2);
    Q << 1, 0 ,
         0,          1;

    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
    auto dist = ellipsoidDistance(p, P, q, Q);
    
    // 记录开始时间
    auto end = std::chrono::high_resolution_clock::now();

        // 计算运行时间
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "dist: " << dist << std::endl;
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    

    return 0;
}
```
output:
```bash
root@Versatrek:/home/Workspace/mesh_planner/test/cpp/build# ./ellipDistCMU 
Time: 56 microseconds
Time: 20 microseconds
Time: 49 microseconds
Time: 70 microseconds
Time: 28 microseconds
Time: 48 microseconds
Time: 17 microseconds
dist: -3
-0
Time taken by function: 345 microseconds
```

# Optimization  
## EigenSlover ---> Eigen::SelfAdjointEigenslover 
because $B, C, B^{-\frac{1}{2}}CB^{-\frac{1}{2}}$ are sym... 
```cpp
Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverB(B);
Eigen::VectorXd LambdaB = eigensolverB.eigenvalues();
Eigen::MatrixXd hatB = eigensolverB.eigenvectors();

    
Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverC(C);
Eigen::VectorXd LambdaC = eigensolverC.eigenvalues();
Eigen::MatrixXd hatC = eigensolverC.eigenvectors(); 

Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolverBinvsqrt_C_Binvsqrt(Binvsqrt_C_Binvsqrt);  
Eigen::VectorXd LambdaQsqrt = eigenSolverBinvsqrt_C_Binvsqrt.eigenvalues().cwiseSqrt();
Eigen::MatrixXd hatQ = eigenSolverBinvsqrt_C_Binvsqrt.eigenvectors();
Eigen::MatrixXd Binvsqrt_hatQ_LambdaQsqrt_hatQT = Binvsqrt * hatQ * LambdaQsqrt.asDiagonal() * hatQ.transpose();
```

## CMakeLists.txt complizer optim 
```cmake  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
```


# Bug  
## Slover Linear System  
- `.ldlt()` only suitable for $A > 0$ && $A^T = A$
- others attention TODO: [REF-LINK](https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html) 
    - `PartialPivLU` need inv condition
    - `FullPivLU` slow
    - `HouseholderQR` fast
    - `ColPivHouseholderQR` fast and good
    - `FullPivHouseholderQR` slow ...
    - `CompleteOrthogonalDecomposition`
    - `LLT` condition

## matrix.dialog()!!!!
```cpp
// OKOKOKOK 
Eigen::EigenSolver<Eigen::MatrixXd> eigensolverM1(M1);
double lambda = eigensolverM1.pseudoEigenvalueMatrix().diagonal().real().minCoeff(); 
```
the eigenvalue result of `eigensolverM1.pseudoEigenvalueMatrix().real().minCoeff()` maybe complex number, so the result will consider the undiagnum, so the result will be `0`, thus error occured.   

## python  matrix np mult  
```python
# @ : matrix mult
# * : elem cwise mult
def computeEllipsoidP(R, S):
    return R @ S @ np.transpose(R)  # ok    matrix mult
    #return R * S * np.transpose(R)  # error cwise  mult
```

# TODO
## linear system issus
//In my experiement, the speed of `fullPivLu` is more faster than `ColPivHouseholderQR`, it's against the compariation [official document](https://eigen.tuxfamily.org/dox/group__TopicLinearAlgebraDecompositions.html)  
