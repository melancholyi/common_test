/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2024-12-17 04:13:31
 * @FilePath: /test/CPP/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

inline Eigen::MatrixXd invDiag(Eigen::MatrixXd& mat){
    Eigen::MatrixXd temp = mat;
    temp.diagonal().array() = mat.diagonal().array().inverse();
    return temp;
}

inline Eigen::MatrixXd invSqrtDiag(Eigen::MatrixXd& mat){
    Eigen::MatrixXd temp = mat;
    temp.diagonal().array() = mat.diagonal().array().inverse().sqrt();
    return temp;
}

    inline int yaw2idx(const double& yaw, double yawResInv){
        return static_cast<int>(std::floor(yaw * yawResInv));
    }
    inline double idx2yaw(const int& idx, double yawres){
        return (double)(idx) * yawres + 0.5 * yawres;
    }

// Function to compute the distance between two ellipsoids
Eigen::VectorXd ellipsoidDistance(const Eigen::MatrixXd& b, const Eigen::MatrixXd& B, const Eigen::MatrixXd& c, const Eigen::MatrixXd& C) {  
    //! dim
    int q = B.cols();
    std::cout << "q:" << q << std::endl;

    //! Step 1: Eigen decomposition of B and C. GET \hat{B}, \Lambda_{B} and \hat{C}, \Lambda_{C}
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverB(B);
    Eigen::MatrixXd LambdaB = eigensolverB.pseudoEigenvalueMatrix();
    Eigen::MatrixXd hatB = eigensolverB.pseudoEigenvectors();
    std::cout << "LambdaB:\n" << LambdaB << std::endl;
    std::cout << "hatB:\n" << hatB << std::endl;

    
    Eigen::EigenSolver<Eigen::MatrixXd> eigensolverC(C);
    Eigen::MatrixXd LambdaC = eigensolverC.pseudoEigenvalueMatrix();
    Eigen::MatrixXd hatC = eigensolverC.pseudoEigenvectors(); 
    std::cout << "LambdaC:\n" << LambdaC << std::endl;
    std::cout << "hatC:\n" << hatC << std::endl;


    //! Step 2: Compute B^(1/2), B^(-1/2), and B^(-1), C^(-1)
    Eigen::MatrixXd Bsqrt = hatB * LambdaB.cwiseSqrt() * hatB.transpose(); // B^(1/2)
    Eigen::MatrixXd Binvsqrt = hatB * invSqrtDiag(LambdaB) * hatB.transpose(); //B^(-1/2)
    Eigen::MatrixXd Binv = hatB * invDiag(LambdaB) * hatB.transpose(); //B^(-1)
    Eigen::MatrixXd Cinv = hatC * invDiag(LambdaC) * hatC.transpose(); //C^(-1)
    std::cout << "Bsqrt:\n" << Bsqrt << std::endl;
    std::cout << "Binvsqrt:\n" << Binvsqrt << std::endl;
    std::cout << "Binv:\n" << Binv << std::endl;
    std::cout << "Cinv:\n" << Cinv << std::endl;
    

    //! Step 3: compute \tilde{C} and \tilde{c}  
    Eigen::MatrixXd tilde_C = Bsqrt * Cinv * Bsqrt;
    std::cout << "tilde_C:\n" << tilde_C << std::endl;

    Eigen::MatrixXd Binvsqrt_C_Binvsqrt = Binvsqrt * C * Binvsqrt;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolverBinvsqrt_C_Binvsqrt(Binvsqrt_C_Binvsqrt);  
    Eigen::MatrixXd LambdaQsqrt = eigenSolverBinvsqrt_C_Binvsqrt.eigenvalues().cwiseSqrt().asDiagonal();
    Eigen::MatrixXd hatQ = eigenSolverBinvsqrt_C_Binvsqrt.eigenvectors();
    Eigen::MatrixXd Binvsqrt_hatQ_LambdaQsqrt_hatQT = Binvsqrt * hatQ * LambdaQsqrt * hatQ.transpose();
    // Eigen::VectorXd tilde_c = Binvsqrt_hatQ_LambdaQsqrt_hatQT.ldlt().solve(c - b);
    Eigen::VectorXd tilde_c = Binvsqrt_hatQ_LambdaQsqrt_hatQT.ldlt().solve(c - b);
    std::cout << "Binvsqrt_C_Binvsqrt:\n" << Binvsqrt_C_Binvsqrt << std::endl;
    std::cout << "LambdaQsqrt:\n" << LambdaQsqrt << std::endl;
    std::cout << "hatQ:\n" << hatQ << std::endl;
    std::cout << "Binvsqrt_hatQ_LambdaQsqrt_hatQT:\n" << Binvsqrt_hatQ_LambdaQsqrt_hatQT << std::endl;
    std::cout << "tilde_c:\n" << tilde_c << std::endl;

    

    //! Step 4: M1 and \lambda
    Eigen::MatrixXd M1(2*q, 2*q);
    Eigen::MatrixXd Iq = Eigen::MatrixXd::Identity(q,q);
    M1 << tilde_C, -Iq,
          -1 * tilde_c * tilde_c.transpose(), tilde_C;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverM1(M1);
    double lambda = eigensolverM1.eigenvalues().minCoeff();   
    std::cout << "eigensolverM1.eigenvalues():\n" << eigensolverM1.eigenvalues() << std::endl;  
    std::cout << "M1:\n" << M1 << std::endl;
    std::cout << "lambda:\n" << lambda << std::endl;
    

    //! Step 5: M2 and compute \alpha, tilde_b
    Eigen::MatrixXd Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt = Binvsqrt * (lambda * Iq - tilde_C) * Bsqrt;
    Eigen::VectorXd alpha = Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt.ldlt().solve(c -b);
    Eigen::VectorXd tilde_b = -lambda * Binvsqrt * alpha;
    Eigen::MatrixXd M2(2*q, 2*q);
    M2<<Binv, -Iq,
        -tilde_b*tilde_b.transpose(), Binv;
    std::cout << "Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt:\n" << Binvsqrt_lambdaIq_minus_tildeC_Binvsqrt << std::endl;
    std::cout << "alpha:\n" << alpha << std::endl;
    std::cout << "tilde_b:\n" << tilde_b << std::endl;
    std::cout << "M2:\n" << M2 << std::endl;
    

    //! Step 6: mu
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolverM2(M2);
    double mu = eigensolverM2.eigenvalues().minCoeff(); 
    std::cout << "mu:\n" << mu << std::endl;
    std::cout << "eigensolverM2.eigenvalues():\n" << eigensolverM2.eigenvalues() << std::endl;

    //! Step 7: compute d* 
    Eigen::MatrixXd mu_Iq_Binv = mu * Iq - Binv;
    Eigen::VectorXd dstar = mu_Iq_Binv.ldlt().solve(-mu * lambda * alpha);
    std::cout << "mu_Iq_Binv:\n" << mu_Iq_Binv << std::endl;
    std::cout << "dstar:\n" << dstar << std::endl;


    return dstar;









    // Eigen::MatrixXd B = P.llt().matrixL();
    // Eigen::MatrixXd C = Q.llt().matrixL();
    // Eigen::MatrixXd BC = B * C.inverse() * B;

    // Eigen::MatrixXd M1 = Eigen::MatrixXd::Identity(2 * p.rows(), 2 * p.rows());
    // M1.block(0, p.rows(), p.rows(), p.rows()) = -BC;
    // M1.block(p.rows(), 0, p.rows(), p.rows()) = -M1.block(0, p.rows(), p.rows(), p.rows()).transpose();

    // Eigen::VectorXd b = B * (q - p);
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(M1);
    // Eigen::VectorXd alpha = eigensolver.eigenvalues().real();
    // double lambda = alpha.minCoeff();

    // Eigen::MatrixXd M2 = Eigen::MatrixXd::Identity(p.rows(), p.rows());
    // M2 -= B.inverse() * b * b.transpose() * B.inverse();
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver2(M2);
    // Eigen::VectorXd beta = eigensolver2.eigenvalues().real();
    // double mu = beta.minCoeff();

    // Eigen::VectorXd dStar = -mu * (B.inverse() * (lambda * B * (q - p)));
    // return dStar.norm();
}

// 假设INPUT_DIM是一个已知的维度
const int INPUT_DIM = 3;

// 模板类BGKInterface的定义
template<int INPUT_DIM, int OUTPUT_DIM, typename T>
class BGKInterface {
public:
    using InputType = Eigen::Matrix<T, -1, INPUT_DIM, Eigen::RowMajor>;
    using OutputType = Eigen::Matrix<T, -1, OUTPUT_DIM, Eigen::RowMajor>;

    BGKInterface(double kLen, double kScalar)
        : kernelLen_(kLen), kernelScaler_(kScalar), isRcvTrainData_(false) {}

    // 打印矩阵的函数
    void print(const InputType& matrix) {
        std::cout << matrix << std::endl;
    }

private:
    double kernelLen_;
    double kernelScaler_;
    bool isRcvTrainData_;
};

//   struct PointsWithVar{
//       Eigen::Matrix3Xd points;
//       Eigen::Matrix3Xd variance;
//       Eigen::Vector3d centroid;  
//       int count;  

//       PointsWithVar(int size){
//         points.resize(3, size); 
//         points.fill(0);
//         variance.resize(3, size); 
//         variance.fill(0);
//         centroid.fill(0);
//         count = 0;
//       }
//   };
const double LARGE_VAR = 100;
struct PointsWithVar{
    Eigen::Matrix3Xd points;
    Eigen::Matrix3Xd variance;
    Eigen::Vector3d centroid;  
    int count;  

    PointsWithVar(int size){
    points.resize(3, size); points.fill(0);
    variance.resize(3, size); variance.fill(LARGE_VAR); 
    centroid.fill(0);
    count = 0;
    }
};

void transformVoxel(){
    // 假设voxel是一个3xN的矩阵，其中N是voxel中的点数
    const int N = 100; // 点的数量
    Eigen::MatrixXd voxel(3, N);
    voxel.setRandom(); // 用随机数填充voxel
    voxel *= 100;
    // std::cout <<  "voxel:\n" << voxel << std::endl << std::endl;

    // 计算均值 and 计算协方差矩阵
    Eigen::Vector3d mean = voxel.rowwise().mean();
    std::cout << "mean:\n" << mean << std::endl << std::endl;
    auto diff = voxel.leftCols(N).colwise() - mean;  
    Eigen::Matrix3d covariance = diff * diff.transpose() / (N - 1);  

    // 计算特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covariance);
    if (eigensolver.info() != Eigen::Success) abort();
    Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
    Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();

    // 输出原始特征值和特征向量
    std::cout << "Original Eigenvalues:\n" << eigenvalues << std::endl << std::endl;
    std::cout << "Original Eigenvectors:\n" << eigenvectors << std::endl << std::endl;

    // 定义旋转矩阵和平移向量
    Eigen::Matrix3d rotation;
    rotation = Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()) *
                Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitY()) *
                Eigen::AngleAxisd(M_PI / 3, Eigen::Vector3d::UnitZ());
    Eigen::Vector3d translation(1, 2, 3);

    std::cout << "rotation:\n" << rotation << std::endl << std::endl;
    // 应用旋转和平移变换
    Eigen::MatrixXd transformed_voxel = (rotation * voxel).colwise() + translation;
    

    // 计算变换后的均值
    Eigen::Vector3d new_mean = transformed_voxel.rowwise().mean();
    std::cout << "new_mean:\n" << new_mean << std::endl << std::endl;
    auto new_diff = transformed_voxel.leftCols(N).colwise() - new_mean;  
    auto new_covariance = new_diff * new_diff.transpose() / (N - 1);  
    std::cout << "new_covariance:\n" << new_covariance << std::endl << std::endl;

    // 计算变换后的特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> new_eigensolver(new_covariance);
    if (new_eigensolver.info() != Eigen::Success) abort();
    Eigen::Matrix3d new_eigenvectors = new_eigensolver.eigenvectors();
    Eigen::Vector3d new_eigenvalues = new_eigensolver.eigenvalues();

    // 输出变换后的特征值和特征向量
    std::cout << "Transformed Eigenvalues:\n" << new_eigenvalues << std::endl << std::endl;
    std::cout << "Transformed Eigenvectors:\n" << new_eigenvectors << std::endl << std::endl;

    //-----------------------direct transform  
    auto mean_direct = rotation * mean + translation;
    auto cov_direct = rotation * covariance;
    // auto eigenval_direct = rotation
    Eigen::Matrix3d eigenvec_direct =  rotation * eigenvectors;  
    std::cout << "----------------direct---------------" << std::endl;
    std::cout << "direct new_mean:\n" << mean_direct << std::endl << std::endl;
    std::cout << "direct new_covariance:\n" << cov_direct << std::endl << std::endl;

    // std::cout << "Transformed Eigenvalues:\n" << new_eigenvalues << std::endl << std::endl;//bu bian
    std::cout << "direct Transformed Eigenvectors:\n" << eigenvec_direct << std::endl << std::endl;

    
}

template<typename T>
class TypeTest{
    public:
    TypeTest(T data){
        if constexpr (std::is_same_v<T, double>) {
            // dtype_ = torch::kFloat64;
            std::cout << "double type" << std::endl;
          } else if constexpr (std::is_same_v<T, float>) {
            // dtype_ = torch::kFloat32;
            std::cout << "float type" << std::endl;
          } else {
            static_assert("T must be float or double");
          }
    }
};


int main() {
    // 创建一个原始数组
    std::vector<double> predX = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // 使用Eigen::Map将原始数组映射到Eigen矩阵
    auto temp_predX = Eigen::Map<BGKInterface<INPUT_DIM, 1, double>::InputType>(predX.data(), predX.size() / INPUT_DIM, INPUT_DIM);

    // 创建BGKInterface对象
    BGKInterface<INPUT_DIM, 1, double> bgkInterface(1.0, 1.0);

    // 打印映射后的矩阵
    bgkInterface.print(temp_predX);

    //222222222222222222222222222222222222222222222

     // 使用 Eigen::Matrix<double, 3, -1> 定义一个 3xN 的矩阵，N 是动态的
    Eigen::Matrix<double, 3, -1> matrix3xN;
    // 由于 -1 表示动态列，我们可以在运行时设置列数
    matrix3xN.resize(3, 4); // 设置为 3x4 矩阵
    matrix3xN << 1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12;
    std::cout << "3xN Matrix (resize to 3x4):\n" << matrix3xN << std::endl;
    matrix3xN.conservativeResize(3, 2);
    std::cout << "Nx3 Matrix (resize to 3x4):\n" << matrix3xN << std::endl;

    // 使用 Eigen::Matrix<double, 3, Eigen::Dynamic> 定义一个 3xN 的矩阵，N 是动态的
    Eigen::Matrix<double, 3, Eigen::Dynamic> matrix3xDynamic;
    // 同样，我们可以在运行时设置列数
    matrix3xDynamic.resize(3, 5); // 设置为 3x5 矩阵
    matrix3xDynamic << 1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15;
    matrix3xDynamic.conservativeResize(3, 3);  
    std::cout << "3xDynamic Matrix (resize to 3x5):\n" << matrix3xDynamic << std::endl;



    ///////////////////////////////////////////////////////////
    // Define a 3x3 real symmetric matrix
    Eigen::Matrix3d A;
    A << 1, 2, 3,
         2, 4, 5,
         3, 5, 6;

    // Create the solver
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(A);

    // Check if the computation was successful
    if (eigensolver.info() == Eigen::Success) {
        // Retrieve the eigenvalues
        Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();

        // Retrieve the eigenvectors
        Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();

        // Print the results
        std::cout << "Eigenvalues:\n" << eigenvalues << std::endl;
        std::cout << "Eigenvectors:\n" << eigenvectors << std::endl;
    } else {
        std::cerr << "Eigenvalue decomposition failed." << std::endl;
    }

    ///////////////////////////////////////////////////////////////
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    PointsWithVar points(5);
    std::cout << points.centroid << std::endl;
    std::cout << points.points << std::endl;


    ///////////////////////////////////////////////
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    int size = 64;
    int num = 63 + size * 2;    
    int count =  (num + size)/size;
    
    auto step = size * ((38+70 + size)/size);
    std::cout << "step" << step << std::endl ;


    /////////////////////////////////////////
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    Eigen::Matrix<double, 3, Eigen::Dynamic> mat3X;
    // 同样，我们可以在运行时设置列数
    mat3X.resize(3, 5); // 设置为 3x5 矩阵
    mat3X << 1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15;
    std::cout << mat3X.size() << " " <<  mat3X.cols() << std::endl;
    Eigen::Vector3d mean = mat3X.leftCols(3).rowwise().sum() / 3;
    std::cout << mean << std::endl;
    
    //---------------------------------------
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    

    PointsWithVar points_(5);
    points_.points << 1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15;
    points_.count = 5;
    int ptcnt = 5;
    auto mean_ = points_.points.leftCols(ptcnt).rowwise().sum() / ptcnt;
    auto diff = points_.points.leftCols(ptcnt).colwise() - mean_;
    auto cov_ = diff * diff.transpose() / (ptcnt - 1);
    std::cout << "mean:\n" << mean_ << std::endl;
    std::cout << "diff:\n" << diff << std::endl;
    std::cout << "cov_:\n" << cov_ << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigenSlover_;
    eigenSlover_.compute(cov_);
    auto eigen_value = eigenSlover_.eigenvalues();
    auto eigen_vector = eigenSlover_.eigenvectors();
    std::cout << "eigen values: \n" << eigen_value << std::endl;
    std::cout << "eigen vector: \n" << eigen_vector << std::endl;

    Eigen::EigenSolver<Eigen::Matrix3d> eig;
    eig.compute(cov_);
    auto eigenvalMat = eig.pseudoEigenvalueMatrix();
    auto eigenvectorMat = eig.pseudoEigenvectors();
    std::cout << "svd eigen values: \n" << eigen_value << std::endl;
    std::cout << "svd eigen vector: \n" << eigen_vector << std::endl;
    std::cout << "svd eigen values(min): \n" << eigen_value(0) << std::endl;
    std::cout << "svd eigen vector(min): \n" << eigen_vector.col(0) << std::endl;
    Eigen::Vector4d minVecVal_;
    minVecVal_ << eigen_value(0), eigen_vector(0,0), eigen_vector(1,0), eigen_vector(2,0);
    std::cout << "minVecVal_ value && vector: \n" << minVecVal_ << std::endl;

    //--dotransform  
    points_.points.conservativeResize(3, 10);

    Eigen::Matrix3Xd pts_new;
    pts_new.resize(3,5);
    pts_new << 1, 2, 3, 4, 5,
                      6, 7, 8, 9, 10,
                      11, 12, 13, 14, 15;
    for(size_t i = 0 ; i < 3; i++){
          points_.points.col(points_.count++) = pts_new.col(i);  
    }
    std::cout << points_.count++ << std::endl;
    int iold = 5, iinc = iold, inew = iold + iinc;
    std::cout << inew << std::endl;
    
    ///////////////////////////////////////////////////
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    transformVoxel();
    ///////////////////////////////////////////////////
    std::cout << "\n!!!!! NEW NEW NEW !!!!!" << std::endl;
    Eigen::Vector3i v1(1, 2, 3);
    Eigen::Vector3i v2(4, 5, 6);
    std::cout << "Vector3i v1+v2 : " << v1+v2 << std::endl;
    // std::cout << "Vector3i v1 * v2 : " << v1 / v2 << std::endl;

    Eigen::Array3i a1(1, 2, 3);
    // 元素级加法（仅适用于 Array）
    Eigen::Array3i a2(4, 5, 6);
    std::cout << "Array 元素级加法: " << a1 + a2 << std::endl;
    std::cout << "Array 元素级mult: " << a1 * a2 << std::endl;
    std::cout << "Array 元素级div: " << a2 / a1 << std::endl;

    Eigen::Vector3i v5 = a2 / a1;
    std::cout << "v5: " << v5 << std::endl;

    ////////////////////////////////////////
    using BoundaryPairType = std::pair<Eigen::Array3d, Eigen::Array3d>;
    class VoxelMap{
        public:
        Eigen::Array3d res_;    //map resolution, maybe extend to un-uniform   
        Eigen::Array3d invRes_; // inverse map resulution  
        
        BoundaryPairType boundary_;
        Eigen::Array3i ijkMulti_;  //index multiplication

        VoxelMap(const Eigen::Array3d &res, const Eigen::Array3d &bMin, const Eigen::Array3d &bMax):
        res_(res), invRes_(Eigen::Array3d::Ones()/res_){  
          boundary_ = std::make_pair(bMin, bMax);   
          Eigen::Array3i voxel_num = ((boundary_.second - boundary_.first) * invRes_).cast<int>();  
          std::cout << "voxel_num:\n" <<  voxel_num << std::endl;
          ijkMulti_(0) = 1;
          ijkMulti_(1) = voxel_num(0);
          ijkMulti_(2) = voxel_num(0) * voxel_num(1);  
        }
    };
    Eigen::Vector3d res(0.5, 0.5, 0.5);
    Eigen::Vector3d b_min(-5.3,-1.3, -2.1);
    Eigen::Vector3d b_max(5.3, 5.2, 5.3);
    VoxelMap voxel_map(res, b_min, b_max);
    std::cout << "inv res:\n" <<  voxel_map.invRes_ << std::endl;
    std::cout << "ijkMulti_:\n" <<  voxel_map.ijkMulti_ << std::endl;  
    
    Eigen::Matrix3Xd temp;
    temp.resize(3,5);
    temp << 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10,
            11, 12, 13, 14, 15;
    std::cout << "temp:\n" <<  temp << std::endl;
    temp.conservativeResize(3,0);
    std::cout << "temp:\n" <<  temp << std::endl;
    temp.conservativeResize(3,4);
    temp << 1, 2, 3, 4,
            6, 7, 8, 9,
            11, 12, 13, 14;
    std::cout << "temp:\n" <<  temp << std::endl;

    Eigen::Array3d num1(-4.2, -3.5, 1.6);
    std::cout << "num1.cast<int>:\n" <<  num1.cast<int>() << std::endl;
    Eigen::Array3d num2(-4.6,  3.5, 1.1);
    std::cout << "num1.cast<int>:\n" <<  num2.cast<int>() << std::endl;

    std::cout << "num1.floor:\n" <<  num1.floor() << std::endl; 
    std::cout << "num1.ceil:\n" <<  num1.ceil() << std::endl;    

    // neg floor
    // pos ceil

    Eigen::MatrixXd arrayTest1, arrayTest2;
    arrayTest1.resize(5,2);
    arrayTest2.resize(5,2);
    arrayTest1 << 0,1,2,3,4,
                  5,6,7,8,9;

    arrayTest2 << 10,11,12,13,14,
                  15,16,17,18,19;
    
    Eigen::MatrixXd arrayTest =  arrayTest1.array()/arrayTest2.array();
    std::cout << "arrayTest:\n" << arrayTest << std::endl;

    //======compute ellipsoid distance
    Eigen::VectorXd p = Eigen::Vector2d(1, 2);
    Eigen::MatrixXd P; P.resize(2,2);
    P << 1, 0,
         0, 1;   

    Eigen::VectorXd q = Eigen::Vector2d(6, 2);
    Eigen::MatrixXd Q; Q.resize(2,2);
    Q << 3, 0 ,
         0,          1;

    auto dist = ellipsoidDistance(p, P, q, Q);
    std::cout << "dist: " << dist << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////
    double res1 = 0.1;
    double res_inv = 1 / res1;  
    double R = 5.0;
    double L = R/sqrt(2);
    std::cout << "L: " << L << std::endl;
    double robotx = 4, roboty = -2;
    std::cout << "robotx - L :" << robotx - L << "\t robotx + L :" << robotx + L << std::endl;
    std::cout << "(robotx - L) * res_inv :" << (robotx - L) * res_inv << "\t (robotx + L) * res_inv :" << (robotx + L) * res_inv << std::endl;
    std::cout << "std::floor((robotx - L) * res_inv) :" << std::floor((robotx - L) * res_inv) << "\t std::floor((robotx + L) * res_inv) :" << std::floor((robotx + L) * res_inv) << std::endl;
    
    std::cout << std::endl;
    
    std::cout << "roboty - L :" << roboty - L << "\t roboty + L :" << roboty + L << std::endl;
    std::cout << "(roboty - L) * res_inv :" << (roboty - L) * res_inv << "\t (roboty + L) * res_inv :" << (roboty + L) * res_inv << std::endl;
    std::cout << "std::floor((roboty - L) * res_inv) :" << std::floor((roboty - L) * res_inv) << "\t std::floor((roboty + L) * res_inv) :" << std::floor((roboty + L) * res_inv) << std::endl;
    
    std::cout << std::endl;

    for(int i = std::floor((robotx - L) * res_inv); i <= std::floor((robotx + L) * res_inv); i++){
        std::cout << "index " << i << std::endl;
        for(int j = std::floor((roboty - L) * res_inv); j <= std::floor((roboty + L) * res_inv); j++){
            std::cout << j << " " ;
        }     
        std::cout << std::endl;
    
    }

    ////////////////////////////////////////////////////////////////////////////
    using Gaussian3DType = std::pair<Eigen::Vector3d, Eigen::Matrix3d>;
    Eigen::Vector3d mean_three(0.1111,0.2222,0.1234567);
    Eigen::Matrix3d cov_three = Eigen::Matrix3d::Identity();
    Gaussian3DType mean_cov = {mean_three, cov_three};
    std::cout << "before mean_cov.first(2): " << mean_cov.first(2) << std::endl;
    mean_cov.first(2) = std::round(mean_cov.first(2) * 1000) / 1000.0;
    std::cout << "after  mean_cov.first(2): " << mean_cov.first(2) << std::endl;


    /////////////////////////////////////////////////////////////////////////
    std::vector<double> vecData;
    for(int i = 0 ; i< 10 ; i ++){
        vecData.push_back(double(i));
    }
    Eigen::MatrixXd matData; matData.resize(3,3);
    matData = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(vecData.data(), 3, 3);
    std::cout << "matData:\n" << matData << std::endl;
    for(int i = 0 ; i< 10 ; i ++){
        vecData[i] = 2*i;
    }
    std::cout << "matData:\n" << matData << std::endl;
    auto matData2 = Eigen::Map<Eigen::Matrix3d>(vecData.data(), 3, 3);
    std::cout << "matData2:\n" << matData2 << std::endl;
    
    ///////////////////////////////////////////////////////////////
    Eigen::MatrixXd divtest;
    arrayTest1.resize(10,1);
    arrayTest1 << 0,1,2,3,4,
                  5,6,7,8,9;
    std::cout << "1/arrayTest1.array():\n" << 1/arrayTest1.array() << std::endl; 

    //////////////////////////////////////////////////////////
    Eigen::Matrix3d covZero = Eigen::Matrix3d::Zero();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> slover(covZero);
    Eigen::Vector3d evals = slover.eigenvalues();
    Eigen::Matrix3d evecs = slover.eigenvectors();
    std::cout << "evals:\n" << evals << std::endl;
    std::cout << "evecs:\n" << evecs << std::endl;

    //////////////////////////////////////////////////////////
    double yaw_res = 0.1;
    double yawres_inv = 1/yaw_res;

    
    Eigen::Vector2d boundYaw;
    boundYaw(0) = static_cast<int>(std::ceil(-M_PI*yawres_inv));
    boundYaw(1) = static_cast<int>(std::floor(M_PI*yawres_inv));
    
    std::cout << "boundYaw:" << boundYaw.transpose() << " points:" << boundYaw(1) - boundYaw(0) + 1  << std::endl;
    


    // 测试用例
    std::vector<double> testYaws{-3.14, -2.50, -1.99, -1.57, 0.0, 0.99, 1.09, 1.57, 2.1 ,3.14};
    int numTests = testYaws.size();

    std::cout << "Testing yaw2idx and idx2yaw:" << std::endl;
    for (int i = 0; i < numTests; ++i) {
        double yaw = testYaws[i];
        int idx = yaw2idx(yaw, yawres_inv);
        double convertedYaw = idx2yaw(idx, yaw_res);

        std::cout << "Yaw: " << yaw << " -> Index: " << idx
                  << " -> Converted Yaw: " << convertedYaw << std::endl;
    }

    ////////////////////////////////////////////////
    Eigen::Matrix3d vaTest1;
    vaTest1 <<  1, 2, 3,
                4, 5, 6,
                7, 8, 9;
    Eigen::Matrix3d vaTest2 = vaTest1 * 2;
    Eigen::Matrix3d vaTestAdd1 = vaTest1.array() * vaTest2.array();
    Eigen::Matrix3d vaTestAdd2 = (vaTest1.array() * vaTest2.array()).array();  
    std::cout << "vaTestAdd1:\n" << vaTestAdd1 << std::endl;
    std::cout << "vaTestAdd2:\n" << vaTestAdd2 << std::endl;

    ////////////////////////////////////////////////
    using RowMajorOutputType = Eigen::Matrix<double, -1, -1, Eigen::RowMajor>;
    using OutputType = Eigen::Matrix<double, -1, 2>;
    std::vector<double> trainYvec;
    for(int i = 0 ; i < 12; i++){
        trainYvec.push_back(i);
    }
    OutputType matY = Eigen::Map<const RowMajorOutputType>(trainYvec.data(), trainYvec.size() / 2, 2);
    std::cout << "matY: \n" << matY << std::endl;
    for(int i = 0 ; i < 12; i++){
        trainYvec[i] = 2*i;
    }
    trainYvec.clear();
    std::cout << "matY: \n" << matY << std::endl;

    //////////////////////////////////////////////////////////////
    static double M2PI = 2 * 3.1415926f;
    Eigen::Matrix3d distMat;
    distMat <<  1, 2, 3,
                4, 5, 6,
                7, 8, 9;
    auto kernel = (-M2PI * (distMat * M2PI).array().sin()) * (1.0f - distMat.array()) / 3;



    ////////////////////////////////////////////////////////////
    Eigen::Matrix3d distMat_T = distMat.transpose();
    std::cout << "distMat_T:\n" << distMat_T << std::endl;


    Eigen::MatrixX3d matX3dtest;
    std::cout << matX3dtest.cols() <<  std::endl;

    /////////////////////////////////////////////////
    for(int i = 0 ; i < 5 ; i++){
        for(int j = 0 ; j < 5 ; j++){
            std::cout << "i:" << i << " j:" << j << std::endl;
            bool is_se2obs_added = false;
            for(int k = 0 ; k < 5 ; k++){
                if(k == 1){
                    is_se2obs_added = true;
                } 
                if(!is_se2obs_added){
                    std::cout << "k:" << k << std::endl;
                }
            }
            std::cout << std::endl;
        
        }
    }

    ///////////////////////////////////////////////////////////////
    Eigen::Vector3d ext(1,1,1);
    Eigen::Vector3d goal(10,10,10);
    auto distTemp = (ext - goal).topRows(2).norm();
    std::cout << "distTemp: " << distTemp << std::endl;


    ////////////////////////////////////////////////////////////
    Eigen::Quaterniond quatTest(0.690482, 0, 0, 0.72335);
    Eigen::Vector3d rpy = quatTest.toRotationMatrix().eulerAngles(0, 1, 2);
    std::cout << "yaw: " << rpy(2) << std::endl;


    ////////////////////////////////////////////////////////////
    double yawRes = 0.1;
    double resYawInv_ = 1/yawRes;
    Eigen::Vector2d boundYaw_;
    boundYaw_(0) = static_cast<int>(std::ceil(-M_PI*resYawInv_));  
    boundYaw_(1) = static_cast<int>(std::floor(M_PI*resYawInv_));  
    for(int index_yaw = boundYaw_(0); index_yaw <= boundYaw_(1); index_yaw++){
        std::cout << "index: " << index_yaw << " , yaw:" << index_yaw * yawRes << std::endl;
    }

    auto normAngle = [](double& angle){
        while (angle < -M_PI)
            angle += 2*M_PI;
        while (angle > M_PI)
            angle -= 2*M_PI;
    };

    double angle = 2.7 + 0.5;
    normAngle(angle);
    std::cout << "normAngle: " << angle << std::endl;  


    /////////////////////////////////////////////////////////////
    struct GridHashFunctor {
        inline size_t operator()(const std::pair<int,int>& xy) const {
            std::cout << "int" << std::endl;    
            return size_t( ((xy.first) * long(73856093)) 
                        ^((xy.second) * long(83492791))) 
                        % size_t(1000000000);
        }
        inline size_t operator()(const std::pair<double, double>& xy) const {
            std::cout << "double" << std::endl;
            auto hash1 = std::hash<double>{}(xy.first);
            auto hash2 = std::hash<double>{}(xy.second);
            return size_t( (hash1 * long(73856093)) 
                    ^(hash2* long(83492791))) 
                    % size_t(1000000000);
        }
    };
    GridHashFunctor gridHashFunctor;
    gridHashFunctor(std::make_pair(1,2));
    gridHashFunctor(std::make_pair(1.1,2.2));


    //////////////////////////////////////////////////////////////////////////////////
    TypeTest<float> typeTest1(1.0);
    TypeTest<double> typeTest2(1.0);    

    return 0;
}
