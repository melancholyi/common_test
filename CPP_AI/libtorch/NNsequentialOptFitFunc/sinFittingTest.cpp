/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-03-31 13:09:41
 * @LastEditTime: 2025-03-31 13:18:52
 * @FilePath: /test/CPP_AI/libtorch/NNsequentialOptFitFunc/sinFittingTest.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <matplotlibcpp.h>  // 包含matplotlib-cpp头文件

namespace plt = matplotlibcpp;  // 使用命名空间

// 定义MLP网络
struct MLP : torch::nn::Module {
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    torch::nn::Linear layer3{nullptr};

    MLP() {
        // 初始化网络层并注册
        layer1 = register_module("layer1", torch::nn::Linear(torch::nn::LinearOptions(1, 16)));
        layer2 = register_module("layer2", torch::nn::Linear(torch::nn::LinearOptions(16, 16)));
        layer3 = register_module("layer3", torch::nn::Linear(torch::nn::LinearOptions(16, 1)));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1(x));
        x = torch::relu(layer2(x));
        x = layer3(x);
        return x;
    }
};

int main() {
    // 创建MLP网络
    MLP mlp;

    // 定义优化器和损失函数
    torch::optim::Adam optimizer(mlp.parameters(), torch::optim::AdamOptions(0.001));
    auto loss_fn = torch::nn::MSELoss();

    // 生成训练数据
    std::vector<float> x_data;
    std::vector<float> y_data;
    for (float x = -10; x <= 10; x += 0.1) {
        x_data.push_back(x);
        y_data.push_back(std::sin(x));
    }

    // 转换为Tensor
    auto x_tensor = torch::tensor(x_data).unsqueeze(1);
    auto y_tensor = torch::tensor(y_data).unsqueeze(1);

    // 训练网络
    int epochs = 10000;
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 前向传播
        auto y_pred = mlp.forward(x_tensor);

        // 计算损失
        auto loss = loss_fn(y_pred, y_tensor);

        // 反向传播和优化
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // 打印损失
        if (epoch % 1000 == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }

    // 打印训练后的损失
    auto y_pred = mlp.forward(x_tensor);
    auto final_loss = loss_fn(y_pred, y_tensor);
    std::cout << "Final Loss: " << final_loss.item<float>() << std::endl;

    // 生成测试数据
    std::vector<float> test_x_data;
    std::vector<float> test_y_data;
    for (float x = -10; x <= 10; x += 0.05) {  // 更密集的测试点
        test_x_data.push_back(x);
        test_y_data.push_back(std::sin(x));
    }

    // 转换为Tensor
    auto test_x_tensor = torch::tensor(test_x_data).unsqueeze(1);

    // 预测
    auto test_y_pred = mlp.forward(test_x_tensor);

    // 转换为向量
    std::vector<float> test_y_pred_data(test_y_pred.data_ptr<float>(), test_y_pred.data_ptr<float>() + test_y_pred.numel());

    // 使用matplotlib-cpp进行可视化
    plt::figure_size(1200, 800);
    plt::title("MLP Fitting y = sin(x)");
    plt::xlabel("x");
    plt::ylabel("y");

    // 绘制真实值
    plt::named_plot("True", test_x_data, test_y_data, "b-");

    // 绘制预测值
    plt::named_plot("Pred", test_x_data, test_y_pred_data, "r--");

    // 显示图例
    plt::legend();

    // 显示图形
    plt::show();

    return 0;
}