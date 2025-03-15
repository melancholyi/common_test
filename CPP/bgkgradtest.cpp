#include <iostream>
#include <Eigen/Dense>
#include <cassert>
#include <vector>

using Eigen::MatrixXd;

const int INPUT_DIM = 3; // Example input dimension
const int OUTPUT_DIM = 2; // Example output dimension

class ExampleClass {
public:
    ExampleClass(const MatrixXd& trainX, const MatrixXd& trainY) : trainX_(trainX), trainY_(trainY) {}

    void computeGradients(const MatrixXd& predX, const MatrixXd& pdDist, const MatrixXd& pdkernel, MatrixXd& kbar, MatrixXd& ybar, MatrixXd& grad) {
        int numPred = predX.rows();
        int numTrain = trainX_.rows();

        MatrixXd pd_kbar(numPred, OUTPUT_DIM);
        MatrixXd pd_ybar(numPred, OUTPUT_DIM);
        MatrixXd YOnes = MatrixXd::Ones(trainY_.rows(), trainY_.cols());

        grad.resize(numPred, INPUT_DIM * OUTPUT_DIM);

        for (int input_idx = 0; input_idx < INPUT_DIM; input_idx++) {
            MatrixXd pdDist_pdInputidx = pdDist.block(0, input_idx * numTrain, numPred, numTrain);
            std::cout << "pdDist_pdInputidx:\n" << pdDist_pdInputidx << std::endl;
            assert(pdkernel.rows() == pdDist_pdInputidx.rows() && pdkernel.cols() == pdDist_pdInputidx.cols() && pdDist_pdInputidx.cols() == YOnes.rows());
            pd_kbar = (pdkernel.array() * pdDist_pdInputidx.array()).matrix() * YOnes; // dim: [pred, output_dim]
            pd_ybar = (pdkernel.array() * pdDist_pdInputidx.array()).matrix() * trainY_; // dim: [pred, output_dim]
            assert(pd_kbar.rows() == kbar.rows() && pd_kbar.cols() == kbar.cols());
            assert(pd_ybar.rows() == ybar.rows() && pd_ybar.cols() == ybar.cols());
            grad.block(0, input_idx * OUTPUT_DIM, numPred, OUTPUT_DIM) =
                (pd_kbar.array() * kbar.array() - kbar.array() * pd_ybar.array()).array() /
                (kbar.array() * kbar.array()).array();
            std::cout << "grad:\n" << grad << std::endl;
        }
    }

private:
    MatrixXd trainX_; // Training data
    MatrixXd trainY_; // Training labels
};


class KernelClass {
public:
    KernelClass(double kernelScaler) : kernelScaler_(kernelScaler) {}

    void pdKernelpdDistMat(const MatrixXd& distMat, MatrixXd& pdkernel) {
        pdkernel.resize(distMat.rows(), distMat.cols());
        static double M2PI = 2 * 3.1415926f;

        pdkernel = ((-M2PI * (distMat * M2PI).array().sin()) * (1.0f - distMat.array()) / 3.0f +
                    (2.0f + (distMat * M2PI).array().cos()) * (-1.0f / 3.0f) +
                    distMat.array().cos()) * kernelScaler_;
    }

private:
    double kernelScaler_; // Scaling factor for the kernel
};

int main() {
    // Example training data (5 samples, 3 features)
    MatrixXd trainX(5, 3);
    trainX << 1, 2, 3,
              4, 5, 6,
              7, 8, 9,
              10, 11, 12,
              13, 14, 15;

    // Example training labels (5 samples, 2 outputs)
    MatrixXd trainY(5, 2);
    trainY << 1, 2,
              3, 4,
              5, 6,
              7, 8,
              9, 10;

    // Example prediction data (3 samples, 3 features)
    MatrixXd predX(3, 3);
    predX << 1.5, 2.5, 3.5,
             4.5, 5.5, 6.5,
             7.5, 8.5, 9.5;

    // Example partial derivative of distance matrix (3 rows, 15 columns)
    MatrixXd pdDist(3, 15);
    pdDist << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                  1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
                  3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5;

    // Example kernel partial derivative matrix (3 rows, 5 columns)
    MatrixXd pdkernel(3, 5);
    pdkernel << 0.1, 0.2, 0.3, 0.4, 0.5,
                0.6, 0.7, 0.8, 0.9, 1.0,
                1.1, 1.2, 1.3, 1.4, 1.5;

    // Example intermediate results
    MatrixXd kbar(3, 2);
    kbar << 1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0;

    MatrixXd ybar(3, 2);
    ybar << 0.5, 1.5,
            2.5, 3.5,
            4.5, 5.5;

    // Initialize the class with training data and labels
    ExampleClass example(trainX, trainY);

    // Output matrix for gradients
    MatrixXd grad;

    // Compute the gradients
    example.computeGradients(predX, pdDist, pdkernel, kbar, ybar, grad);

    // Print the result
    std::cout << "Gradient Matrix:\n" << grad << std::endl;

    std::vector<int> index;
    for(int i = 0; i < OUTPUT_DIM; i++){
        for(int j = 0; j < INPUT_DIM; j++){
            index.push_back(i + j * OUTPUT_DIM);
        }
    }
    Eigen::PermutationMatrix<Eigen::Dynamic> perm(index.size());
    perm.indices() = Eigen::Map<Eigen::VectorXi>(index.data(), index.size());
    grad *= perm;
    std::cout << "grad_order:\n" << grad << std::endl;

    for(auto num : index){
        std::cout << num << " " ;
    }
    std::cout << std::endl;



    // Example distance matrix (3 rows, 5 columns)
    MatrixXd distMat(3, 5);
    distMat << 0.1, 0.2, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1.0,
               1.1, 1.2, 1.3, 1.4, 1.5;

    // Initialize the kernel scaler
    double kernelScaler = 0.5;

    // Initialize the class with the kernel scaler
    KernelClass kernelClass(kernelScaler);

    // Output matrix for the partial derivative of the kernel
    MatrixXd pdkernel2;

    // Compute the partial derivative of the kernel
    kernelClass.pdKernelpdDistMat(distMat, pdkernel2);

    // Print the result
    std::cout << "Partial Derivative of Kernel:\n" << pdkernel2 << std::endl;

    return 0;
}