#include <Eigen/Dense>
#include <open3d/Open3D.h>
#include "open3d/core/CUDAUtils.h"
#include <iostream>
#include <typeinfo>

template<typename TensorType>
class TensorWrapper{
    public:
    TensorWrapper(const TensorType& tensor){
        tensor_ = tensor;
    }

    template<typename T>
    TensorType add(T scalar){
        std::cout << "Type of T: " << __PRETTY_FUNCTION__ << std::endl;
        return tensor_ + scalar;
    }

    TensorType tensor_;

};

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


// 计算加权平均
Eigen::MatrixXd operTensor(const Eigen::MatrixXd& matrix) {
    // std::vector<double> vec;
    // vec.push_back(1.0);
    // // std::cout << decltype(vec[0]) << std::endl;

    // 将 Eigen::Matrix 转换为 Open3D Tensor
    open3d::core::Device device("CUDA:0"); // 使用 GPU
    std::cout << "=====open3d::core::cuda::IsAvailable()" << open3d::core::cuda::IsAvailable() << std::endl;

    open3d::core::Tensor tensor(matrix.data(), // data-begin
                                {matrix.rows(), matrix.cols()}, // shape: {rows, cols}
                                open3d::core::Float64, 
                                device);
    std::cout << "==========tensor: \n" << tensor.ToString() << std::endl;
    std::cout << "==========shape: " << tensor.GetShape().ToString() << std::endl;
    
    //matmul cuda gpu
    compRuntime runtime1;
    for(int i = 0; i < 10000; i++){
        tensor += tensor;
        tensor /= 2;
    }
    std::cout << "runtime1: " << runtime1.getRuntimeMs() << std::endl;
    std::cout << tensor.ToString() << std::endl;

    compRuntime runtime2;
    auto temp_matrix = matrix;
    for(int i = 0; i < 10000; i++){
        temp_matrix *= 2;
        temp_matrix /= 2;
    }
    std::cout << "runtime2: " << runtime2.getRuntimeMs() << std::endl;
    std::cout << temp_matrix << std::endl;


    int rows = matrix.rows();
    int cols = matrix.cols();

    Eigen::MatrixXd matrix_oped = open3d::core::eigen_converter::TensorToEigenMatrixXd(tensor);

    std::cout << "matrix: \n" << matrix_oped << std::endl;


    TensorWrapper<open3d::core::Tensor> tensor_wrapper(tensor);
    tensor_wrapper = tensor_wrapper.add(false);
    std::cout << tensor_wrapper.tensor_.ToString() << std::endl;




    Eigen::MatrixXd result(rows, cols);


    return result;
}

class Open3DTensorTest{
    public:

    template<typename T>
    Open3DTensorTest(const std::vector<T>& datas, const std::initializer_list<int64_t>& shape){
        open3d::core::Device device = open3d::core::cuda::IsAvailable() ? open3d::core::Device("CUDA:0") : open3d::core::Device("CPU:0");
        tensor_ = open3d::core::Tensor(datas, shape, open3d::core::Dtype::FromType<T>(), device);
        std::cout << "Test device: \n" << device.ToString() << std::endl;
        std::cout << "Test tensor: \n" << tensor_.ToString() << std::endl;
    }

    template <class Derived>
    Open3DTensorTest(const Eigen::MatrixBase<Derived> &matrix){
        // if(rowMajor){
            tensor_ = open3d::core::eigen_converter::EigenMatrixToTensor(matrix);
            open3d::core::Device device = open3d::core::cuda::IsAvailable() ? open3d::core::Device("CUDA:0") : open3d::core::Device("CPU:0");
            tensor_.To(device);
        // }else{
        //     typedef typename Derived::Scalar Scalar;
        //     open3d::core::Dtype dtype = open3d::core::Dtype::FromType<Scalar>();
        //     Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
        //             matrix_col_major = matrix;
        //     tensor_ = open3d::core::Tensor(matrix_col_major.transpose().data(), {matrix.rows(), matrix.cols()},
        //                         dtype);
        // }

        

        std::cout << "Test device: \n" << tensor_.IsCUDA() << std::endl;
        std::cout << "Test tensor: \n" << tensor_.ToString() << std::endl;
    }

    open3d::core::Tensor tensor_;

};

void cmpTwoOper(){
    // 创建两个形状为 1000x1000 的张量，填充值为 1.0
    open3d::core::SizeVector shape1 = {10000, 1000};
    open3d::core::SizeVector shape = {1000, 4};
    float fill_value = 1.0f;
    open3d::core::Dtype dtype = open3d::core::Float32;

    // 在 CPU 上创建张量
    open3d::core::Tensor tensor1_cpu = open3d::core::Tensor::Full(shape1, fill_value, dtype);
    open3d::core::Tensor tensor2_cpu = open3d::core::Tensor::Full(shape, fill_value*2, dtype);

    // 在 CUDA 上创建张量
    open3d::core::Device cuda_device("CUDA:0");
    open3d::core::Tensor tensor1_cuda = open3d::core::Tensor::Full(shape1, fill_value, dtype, cuda_device);
    open3d::core::Tensor tensor2_cuda = open3d::core::Tensor::Full(shape, fill_value*2, dtype, cuda_device);

    // CPU 上的计算
    auto start_cpu = std::chrono::high_resolution_clock::now();
    open3d::core::Tensor result_cpu;
    for(int i = 0 ; i< 100 ; i++){
        // result_cpu = tensor1_cpu * tensor2_cpu;  
        // result_cpu /= 2;
        result_cpu = tensor1_cpu.Matmul(tensor2_cpu);
    }
     
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();
    std::cout << "CPU Time: " << duration_cpu << " ms" << std::endl;

    // CUDA 上的计算
    auto start_cuda = std::chrono::high_resolution_clock::now();
    open3d::core::Tensor result_cuda;
    for(int i = 0 ; i< 100 ; i++){
        // result_cuda = tensor1_cuda * tensor2_cuda;  
        // result_cuda /= 2;
        result_cuda = tensor1_cuda.Matmul(tensor2_cuda);
    }
    auto end_cuda = std::chrono::high_resolution_clock::now();
    auto duration_cuda = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();
    std::cout << "CUDA Time: " << duration_cuda << " ms" << std::endl;

    // // eigen  
    // Eigen::MatrixXd matrix1(2000, 4);
    // matrix1.fill(1.0);
    // Eigen::MatrixXd matrix2(2000, 4);
    // matrix2.fill(2.0);
    // auto start_eigen = std::chrono::high_resolution_clock::now();
    // Eigen::MatrixXd result_eigen;
    // for(int i = 0 ; i< 100 ; i++){
    //     // result_eigen = matrix1 * matrix2;  
    //     // result_eigen /= 2;
    //     result_eigen = matrix1 * matrix2.transpose();
    // }
    // auto end_eigen = std::chrono::high_resolution_clock::now();
    // auto duration_eigen = std::chrono::duration_cast<std::chrono::milliseconds>(end_eigen - start_eigen).count();
    // std::cout << "Eigen Time: " << duration_eigen << " ms" << std::endl;
}
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

void covSparse(const O3DTensorType& predX, const O3DTensorType& distMat, O3DTensorType& kernel) {
    kernel.resize(predX.rows(), trainX_.rows());
    kernel = (((2.0f + (distMat * 2.0f * 3.1415926f).array().cos()) * (1.0f - distMat.array()) / 3.0f) +
          (distMat * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * this->kernelScaler_;

    // Clean up for values with distance outside length scale
    // Possible because Kxz <= 0 when dist >= this->kernelLen_
    for (int i = 0; i < kernel.rows(); ++i)
    {
        for (int j = 0; j < kernel.cols(); ++j)
            if (kernel(i,j) < 0.0)
                kernel(i,j) = 0.0f;
    }
  }



int main() {
    // // 创建一个示例矩阵
    // Eigen::MatrixXd matrix(4, 5);
    // matrix << 1, 2, 3, 4, 5,
    //           6, 7, 8, 9, 10,
    //           11, 12, 13, 14, 15,
    //           16, 17, 18, 19, 20,
    //         //   21, 22, 23, 24, 25;
    
    // std::cout << matrix << std::endl;


    // // 计算加权平均
    // Eigen::MatrixXd result = operTensor(matrix);

    // cmpTwoOper();

    // // // 输出结果
    // // std::cout << "Original matrix:\n" << matrix << std::endl;
    // // std::cout << "Weighted average matrix:\n" << result << std::endl;

    std::vector<double> predXvec{1, 2, 3, 4,
                                5, 6, 7, 8,
                                9, 10, 11, 12};
                        
    std::vector<double> trainXvec{1, 2, 3, 4, 
                                5, 6, 7, 8, 
                                9, 10,11, 12, 
                                13, 14, 15, 16, 
                                17, 18, 19, 20};


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
    std::cout << "=====OUT distMat: \n" << distMat.ToString() << std::endl;


    // std::cout << "=====open3d::core::cuda::IsAvailable()" << open3d::core::cuda::IsAvailable() << std::endl;
    // Open3DTensorTest tensorTest(vec, {5,4});
    // Eigen::MatrixXd matrix2(4, 5);
    // matrix2 << 1, 2, 3, 4, 5,
    //           6, 7, 8, 9, 10,
    //           11, 12, 13, 14, 15,
    //           16, 17, 18, 19, 20;
    // Open3DTensorTest tensorTest1(matrix2);


    return 0;
}