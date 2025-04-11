/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-09 06:21:40
 * @LastEditTime: 2025-04-09 07:10:26
 * @FilePath: /test/CPP_AI/libtorch/cmpkFloatxx/cmpkFloatxx.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <torch/torch.h>
#include <iostream>
#include <chrono>

// 定义MLP模型
struct MLP : torch::nn::Module {
    torch::nn::Linear fc1;
    torch::nn::Linear fc2;
    torch::nn::Linear fc3;

    MLP(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1(torch::nn::LinearOptions(input_size, hidden_size)),
          fc2(torch::nn::LinearOptions(hidden_size, hidden_size)),
          fc3(torch::nn::LinearOptions(hidden_size, output_size)) {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }
};

// 训练函数
void trainTest(torch::Dtype dtype) {
    // 设置随机种子
    torch::manual_seed(42);

    // 定义模型、优化器和损失函数
    MLP model(1, 128, 1);
    model.to(torch::kCUDA, dtype);

    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.001));

    // 生成训练数据
    auto x = torch::randn({10000, 1}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));
    std::cout << x.dtype() << " " << x.device() << std::endl;
    auto y = 2 * torch::sin(x) + 4 * torch::cos(x) + torch::pow(x, 2);

    // 训练模型
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int epoch = 0; epoch < 1000; epoch++) {
        auto y_pred = model.forward(x);
        auto loss = torch::mse_loss(y_pred, y);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算运行时间
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Training with " << (dtype == torch::kFloat16 ? "kFloat16" : "kFloat32") << " took " << duration.count() << " ms" << std::endl;
}

int main() {
    // 训练模型并比较运行时间
    trainTest(torch::kFloat16);
    trainTest(torch::kFloat32);

    return 0;
}