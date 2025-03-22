#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include "open3d/core/EigenConverter.h"
#include "open3d/core/CUDAUtils.h"
#include <iostream>
#include <typeinfo>

#include <cmath>

class compRuntime{
    private:
    std::chrono::_V2::system_clock::time_point start_, end_;


    public:
    compRuntime(){
        start_ = std::chrono::high_resolution_clock::now();
    }
    double getRuntimeMs(){
        end_ = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }
};


using O3DTensorType = open3d::core::Tensor;
using O3DDeviceType = open3d::core::Device;
using O3DDType = open3d::core::Dtype;
auto device_ = open3d::core::cuda::IsAvailable() ? O3DDeviceType("CUDA:0") : O3DDeviceType("CPU:0");
auto dtype_ = O3DDType::Float64; 

using ComputeDistMatFunType = std::function<void(const O3DTensorType&, const O3DTensorType&, O3DTensorType&)>;


void computeDistMat(const O3DTensorType& predX, const O3DTensorType& trainX, O3DTensorType& distMat){
    int num_pred = predX.GetShape(0);
    int num_train = trainX.GetShape(0);
    int num_dim = predX.GetShape(1);

    auto predX_broadcast = (predX.Reshape({num_pred, 1, num_dim})).Broadcast({num_pred, num_train, num_dim});
    auto trainX_broadcast = (trainX.Reshape({1, num_train, num_dim})).Broadcast({num_pred, num_train, num_dim});
    auto diff = predX_broadcast - trainX_broadcast;


    distMat = (diff * diff).Sum({2}).Sqrt();

    std::cout << "=====trainX.Broadcast:\n" << predX_broadcast.ToString() << std::endl;
    std::cout << "=====predX_broadcast:\n" << trainX_broadcast.ToString() << std::endl;
    std::cout << "=====diff:\n" << diff.ToString() << std::endl;
    std::cout << "=====diff Squa:\n" << (diff * diff).ToString() << std::endl;
    std::cout << "=====diff Squa Sum:\n" << (diff * diff).Sum({2}).ToString() << std::endl;
    std::cout << "=====distMat:\n" << distMat.ToString() << std::endl;
}

const double M2PI = 2.0 * M_PI;
double kernelLen_ = 2.0;
double kernelScaler_ = 1.0;
void covSparse(const O3DTensorType& distMat, O3DTensorType& kernel, Eigen::MatrixXd& ybar){

    kernel = ((2.0 + (distMat * M2PI / kernelLen_).Cos()) * (1.0 - distMat/kernelLen_) / 3.0 +
                     (distMat * M2PI / kernelLen_).Sin() / M2PI) * kernelScaler_;

    std::cout << "=====kernel:\n" << kernel.ToString() << std::endl;

    //test getItem  

    std::cout << "=====kernel Item to Printf!!!" << std::endl;

    for(int i = 0 ; i < kernel.GetShape(0); i++){
        for(int j = 0; j < kernel.GetShape(1); j++){
            std::cout << kernel[i][j].Item<double>() << " ";
        }
        std::cout << std::endl;
    }

    // kernel's elem is masked with 0.0 if dist > kernelLen_
    kernel = kernel * (kernel.Ge(0.0)).To(dtype_);



    std::cout << "=====kernel masked:\n" << kernel.ToString() << std::endl;
    std::cout << "=====kernel device: " << kernel.GetDevice().ToString() << std::endl;


    
    ybar = open3d::core::eigen_converter::TensorToEigenMatrixXd(kernel);
    std::cout << "=====kernel eigen:\n" << ybar << std::endl;

    // std::vector<double> trainXvec{
    //     1, 2, 3, 4, 
    //     5, 6, 7, 8, 
    //     9, 10,11, 12, 
    //     13, 14, 15, 16, 
    //     17, 18, 19, 20};
    // O3DTensorType trainX = O3DTensorType(trainXvec, {(int)trainXvec.size() / 4, 4}, dtype_, device_);
    // std::cout << "=====kernel @ trainX: \n" << kernel.Matmul(trainX).ToString() << std::endl;

    // // Clean up for values with distance outside length scale
    // // Possible because Kxz <= 0 when dist >= this->kernelLen_
    // for (int i = 0; i < kernel.rows(); ++i)
    // {
    //     for (int j = 0; j < kernel.cols(); ++j)
    //         if (kernel(i,j) < 0.0)
    //             kernel(i,j) = 0.0f;
    // }
  }



  void test(const std::vector<double>& predXvec, const std::vector<double>& trainXvec, Eigen::MatrixXd& ybar){

    O3DDType dtype_ = O3DDType::Float64;
    O3DDeviceType device_ = O3DDeviceType("CUDA:0");
    const int INPUT_DIM = 4;


    O3DTensorType trainX = O3DTensorType(trainXvec, {(int)trainXvec.size() / INPUT_DIM, INPUT_DIM}, dtype_, device_);
    std::cout << "=====trainX: \n" << trainX.ToString() << std::endl;

    O3DTensorType predX = O3DTensorType(predXvec, {(int)predXvec.size() / INPUT_DIM, INPUT_DIM}, dtype_, device_);
    std::cout << "=====predX: \n" << predX.ToString() << std::endl;

    O3DTensorType distMat;

    auto computeDistMatFunc_ = std::bind(&computeDistMat, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    
    // computeDistMat(predX, trainX, distMat);
    computeDistMatFunc_(predX, trainX, distMat);
    std::cout << "=====(distMat/kernelLen_) distMat: \n" << (distMat/kernelLen_).ToString() << std::endl;

    O3DTensorType kernel;
    covSparse(distMat, kernel, ybar);
    std::cout << "=====ybar kernel eigen:\n" << ybar << std::endl;
  }



int main() {
    std::vector<double> predXvec{1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12};
                        
    std::vector<double> trainXvec{1, 2, 3, 4, 
                                5, 6, 7, 8, 
                                9, 10,11, 12, 
                                13, 14, 15, 16, 
                                17, 18, 19, 20};


    
    //==========Test Code================
    Eigen::MatrixXd ybar;
    test(predXvec, trainXvec, ybar);

    return 0;
}