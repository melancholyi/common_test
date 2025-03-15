#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <iostream>
#include <chrono>

using namespace Eigen;
using namespace std::chrono;
double ellipsoidDistance(const VectorXd& b, const MatrixXd& B, const VectorXd& c, const MatrixXd& C) {
    // 计算形状矩阵的平方根
    MatrixXd B_sqrt = B.llt().matrixL();
    MatrixXd C_sqrt = C.llt().matrixL();

    // 初始化单位球面上的点
    VectorXd u = VectorXd::Random(B.rows());
    u = u / u.norm();
    VectorXd v = VectorXd::Random(C.rows());
    v = v / v.norm();

    // 定义距离函数
    auto distance = [&](const VectorXd& u, const VectorXd& v) {
        return (b + B_sqrt * u - (c + C_sqrt * v)).norm();
    };

    // 梯度下降法
    double learning_rate = 0.01;
    double tolerance = 1e-4;
    double prev_distance = distance(u, v);
    double current_distance;

    do {
        // 计算梯度
        VectorXd grad_u = B_sqrt.transpose() * (b + B_sqrt * u - (c + C_sqrt * v)) / prev_distance;
        VectorXd grad_v = -C_sqrt.transpose() * (b + B_sqrt * u - (c + C_sqrt * v)) / prev_distance;

        // 更新 u 和 v
        u = u - learning_rate * grad_u;
        v = v - learning_rate * grad_v;

        // 归一化 u 和 v
        u = u / u.norm();
        v = v / v.norm();

        // 计算新的距离
        current_distance = distance(u, v);

        // 检查收敛性
        if (std::abs(prev_distance - current_distance) < tolerance) {
            break;
        }

        prev_distance = current_distance;
    } while (true);

    return current_distance;
}

int main() {
    Eigen::VectorXd p = Eigen::Vector2d(1, 2);
    Eigen::MatrixXd P; P.resize(2,2);
    P << 1, 0,
         0, 1;   

    Eigen::VectorXd q = Eigen::Vector2d(6, 2);
    Eigen::MatrixXd Q; Q.resize(2,2);
    Q << 3, 0 ,
         0,          1;

    // 记录开始时间
    auto start = high_resolution_clock::now();

    // 计算椭球体之间的距离
    double distance = ellipsoidDistance(p, P, q, Q);
    std::cout << "The distance between the two ellipsoids is: " << distance << std::endl;

    // 记录结束时间
    auto stop = high_resolution_clock::now();

    // 计算运行时间
    auto duration = duration_cast<microseconds>(stop - start);

     std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;

    return 0;
}