#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <matplotlibcpp.h>

namespace plt = matplotlibcpp;

// 定义五次多项式模型
struct PolynomialModel : torch::nn::Module {
    torch::Tensor forward(torch::Tensor x) {
        return a0 + a1 * x + a2 * x.pow(2) + a3 * x.pow(3) + a4 * x.pow(4) + a5 * x.pow(5);
    }

    torch::Tensor a0 = register_parameter("a0", torch::zeros({}));
    torch::Tensor a1 = register_parameter("a1", torch::zeros({}));
    torch::Tensor a2 = register_parameter("a2", torch::zeros({}));
    torch::Tensor a3 = register_parameter("a3", torch::zeros({}));
    torch::Tensor a4 = register_parameter("a4", torch::zeros({}));
    torch::Tensor a5 = register_parameter("a5", torch::zeros({}));
};

void testWithoutTorchNNModule(){
    // 生成训练数据
    auto x = torch::linspace(-M_PI, M_PI, 1000).to(torch::kFloat32);
    auto y_true = 2 * torch::sin(x) + 4 * torch::cos(x) + x.pow(2);
    auto y = y_true.clone();
    y += torch::normal(0, 0.1, y.sizes()).to(torch::kFloat32); // 添加正态分布噪声

    // 初始化五次多项式参数
    auto params = torch::zeros({6}, torch::requires_grad());
    auto optimizer = torch::optim::LBFGS(
        std::vector<torch::Tensor>({params}),
        torch::optim::LBFGSOptions().lr(0.01).max_iter(20).max_eval(30).tolerance_grad(1e-7).tolerance_change(1e-9).history_size(100).line_search_fn("strong_wolfe")
    );

    // 训练模型
    auto closure = [&]() -> torch::Tensor {
        optimizer.zero_grad();
        auto pred = params[0] + params[1] * x + params[2] * x.pow(2) + params[3] * x.pow(3) + params[4] * x.pow(4) + params[5] * x.pow(5);
        auto loss = torch::nn::functional::mse_loss(pred, y);
        loss.backward();
        return loss;
    };

    for (int i = 0; i < 100; i++) {
        optimizer.step(closure);
        auto loss = closure();
        std::cout << "Iteration: " << i + 1 << ", Loss: " << loss.item<float>() << std::endl;
    }

    // 绘制结果
    auto pred = params[0] + params[1] * x + params[2] * x.pow(2) + params[3] * x.pow(3) + params[4] * x.pow(4) + params[5] * x.pow(5);
    std::vector<float> x_vec(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
    std::vector<float> y_true_vec(y_true.data_ptr<float>(), y_true.data_ptr<float>() + y_true.numel());
    std::vector<float> y_vec(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
    std::vector<float> pred_vec(pred.data_ptr<float>(), pred.data_ptr<float>() + pred.numel());

    plt::figure_size(1000, 600);
    plt::plot(x_vec, y_true_vec, "b-");
    plt::scatter(x_vec, y_vec, 1);
    plt::plot(x_vec, pred_vec, "g--");
    plt::grid(true);
    plt::legend();
    plt::show();
}

int main() {
    double noise_std = 0.1; // 噪声标准差
    double lr = 0.001; // 学习率
    int epochs = 100; // 迭代次数
    int sample_num = 1000; // 训练样本数量



    // 生成训练数据
    auto x = torch::linspace(-M_PI, M_PI, sample_num).to(torch::kFloat32);
    auto y_true = 2 * torch::sin(x) + 4 * torch::cos(x) + x.pow(2);
    auto y = y_true.clone();
    y += torch::normal(0, noise_std, y.sizes()).to(torch::kFloat32); // 添加正态分布噪声

    // 初始化模型和优化器
    auto model = std::make_shared<PolynomialModel>();
    auto optimizer = torch::optim::LBFGS(
        model->parameters(),
        torch::optim::LBFGSOptions().lr(lr).max_iter(20).max_eval(30).tolerance_grad(1e-7).tolerance_change(1e-9).history_size(100).line_search_fn("strong_wolfe")
    );

    // 训练模型
    auto closure = [&]() -> torch::Tensor {
        optimizer.zero_grad();
        auto pred = model->forward(x);
        auto loss = torch::nn::functional::mse_loss(pred, y);
        loss.backward();
        return loss;
    };

    for (int i = 0; i < epochs; i++) {
        optimizer.step(closure);
        auto loss = closure();
        std::cout << "Iteration: " << i + 1 << ", Loss: " << loss.item<float>() << std::endl;
    }

    // 绘制结果
    auto pred = model->forward(x);
    std::vector<float> x_vec(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
    std::vector<float> y_true_vec(y_true.data_ptr<float>(), y_true.data_ptr<float>() + y_true.numel());
    std::vector<float> y_vec(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
    std::vector<float> pred_vec(pred.data_ptr<float>(), pred.data_ptr<float>() + pred.numel());

    plt::figure_size(1000, 600);
    plt::named_plot("Ground Truth", x_vec, y_true_vec, "b-");
    plt::scatter(x_vec, y_vec, 1);
    plt::named_plot("Prediction", x_vec, pred_vec, "r--");
    plt::grid(true);
    plt::legend();
    plt::show();

    testWithoutTorchNNModule();

    return 0;
}

