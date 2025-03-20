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
                                {matrix.cols(), matrix.rows()}, // shape: {rows, cols}
                                open3d::core::Float64, 
                                device);

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

int main() {
    // 创建一个示例矩阵
    Eigen::MatrixXd matrix(5, 5);
    matrix << 1, 2, 3, 4, 5,
              6, 7, 8, 9, 10,
              11, 12, 13, 14, 15,
              16, 17, 18, 19, 20,
              21, 22, 23, 24, 25;
    
    std::cout << matrix << std::endl;


    // 计算加权平均
    Eigen::MatrixXd result = operTensor(matrix);

    // // // 输出结果
    // // std::cout << "Original matrix:\n" << matrix << std::endl;
    // // std::cout << "Weighted average matrix:\n" << result << std::endl;


    // std::vector<double> vec{1, 2, 3, 4, 5,
    //     6, 7, 8, 9, 10,
    //     11, 12, 13, 14, 15,
    //     16, 17, 18, 19, 20};

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