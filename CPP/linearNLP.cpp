#include <iostream>
#include <cmath>
#include <Eigen/Dense>

// 双数结构体，用于实现前向模式自动微分
struct Dual {
    double value;      // 函数值
    double derivative; // 导数值

    // 构造函数，初始化双数
    Dual(double v, double d) : value(v), derivative(d) {}

    // 重载加法运算符
    Dual operator+(const Dual& other) const {
        return Dual(value + other.value, derivative + other.derivative);
    }

    // 重载乘法运算符
    Dual operator*(const Dual& other) const {
        return Dual(value * other.value, value * other.derivative + derivative * other.value);
    }

    // 重载正弦函数
    friend Dual sin(const Dual& x) {
        return Dual(std::sin(x.value), std::cos(x.value) * x.derivative);
    }

    // 重载余弦函数
    friend Dual cos(const Dual& x) {
        return Dual(std::cos(x.value), -std::sin(x.value) * x.derivative);
    }

    // 重载正切函数
    friend Dual tan(const Dual& x) {
        return Dual(std::tan(x.value), 1.0 / std::cos(x.value) / std::cos(x.value) * x.derivative);
    }
};

// 差速车辆运动学方程
void dynamics(const Eigen::VectorXd& x, const Eigen::VectorXd& u, Eigen::VectorXd& dxdt, const double wheelbase) {
    dxdt(0) = u(0) * cos(x(2));
    dxdt(1) = u(0) * sin(x(2));
    dxdt(2) = u(0) * (tan(u(1)) / wheelbase);
}

// 一阶线性化函数
void linearizedContinuous(const Eigen::VectorXd& x, const Eigen::VectorXd& u, 
                          Eigen::MatrixXd& F, Eigen::MatrixXd& G, const double wheelbase) {
    // 初始化双数
    Eigen::VectorXd x_dual(3);
    Eigen::VectorXd u_dual(2);
    Eigen::VectorXd dxdt(3);
    for (int i = 0; i < 3; ++i) {
        x_dual(i) = Dual(x(i), 0.0);
    }
    for (int i = 0; i < 2; ++i) {
        u_dual(i) = Dual(u(i), 0.0);
    }

    // 计算对 x 的导数
    for (int i = 0; i < 3; ++i) {
        x_dual(i).derivative = 1.0;
        dynamics(x_dual, u_dual, dxdt, wheelbase);
        F(0, i) = dxdt(0).derivative;
        F(1, i) = dxdt(1).derivative;
        F(2, i) = dxdt(2).derivative;
        x_dual(i).derivative = 0.0;
    }

    // 计算对 u 的导数
    for (int i = 0; i < 2; ++i) {
        u_dual(i).derivative = 1.0;
        dynamics(x_dual, u_dual, dxdt, wheelbase);
        G(0, i) = dxdt(0).derivative;
        G(1, i) = dxdt(1).derivative;
        G(2, i) = dxdt(2).derivative;
        u_dual(i).derivative = 0.0;
    }
}

int main() {
    const double wheelbase = 1.0; // 轮距
    Eigen::VectorXd x(3); // 状态向量 [x, y, theta]
    x << 1.0, 2.0, M_PI / 4;
    Eigen::VectorXd u(2); // 输入向量 [v, w]
    u << 1.0, 0.1;

    Eigen::MatrixXd F(3, 3);
    Eigen::MatrixXd G(3, 2);

    linearizedContinuous(x, u, F, G, wheelbase);

    std::cout << "状态矩阵 F:" << std::endl << F << std::endl;
    std::cout << "控制矩阵 G:" << std::endl << G << std::endl;

    return 0;
}