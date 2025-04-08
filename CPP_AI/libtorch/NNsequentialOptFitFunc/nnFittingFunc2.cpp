// /*
//  * @Author: chasey && melancholycy@gmail.com
//  * @Date: 2025-03-31 13:09:41
//  * @LastEditTime: 2025-04-01 02:10:36
//  * @FilePath: /test/CPP_AI/libtorch/NNsequentialOptFitFunc/sinFittingTest2.cpp
//  * @Description: 
//  * @Reference: 
//  * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
//  */
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <matplotlibcpp.h>  // 包含matplotlib-cpp头文件

// namespace plt = matplotlibcpp;  // 使用命名空间

// torch::nn::Sequential createSequentialModel(int dimInput, int dimHidden, int dimOutput, bool isSoftmax = true) {
//     // Create a sequential model with three layers
//     torch::nn::Sequential model = torch::nn::Sequential({
//         {"linear1", torch::nn::Linear(dimInput, dimHidden)},
//         {"activation1", torch::nn::Tanh()},
//         {"linear2", torch::nn::Linear(dimHidden, dimHidden)},
//         {"activation2", torch::nn::Tanh()},
//         {"output", torch::nn::Linear(dimHidden, dimOutput)}
//     });
//     if (isSoftmax){
//         model->push_back("activation3", torch::nn::Softmax(1));  
//     }
//     return model;
// }


// class ThreeLayerTanhNN : public torch::nn::Module {
//     public:
//     ThreeLayerTanhNN(int dimInput, int dimHidden, int dimOutput, bool isSoftmax = true){
//         linear1_ = register_module("linear1", torch::nn::Linear(dimInput, dimHidden));
//         linear2_ = register_module("linear2", torch::nn::Linear(dimHidden, dimHidden));
//         linear3_ = register_module("linear3", torch::nn::Linear(dimHidden, dimOutput));
//         isSoftmax_ = isSoftmax;
//         if (isSoftmax_){
//             softmax_ = register_module("activation3", torch::nn::Softmax(1));
//         }
//     }


//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(linear1_->forward(x));
//         x = torch::relu(linear2_->forward(x));
//         x = linear3_->forward(x);
//         if (isSoftmax_) {
//             x = softmax_->forward(x);
//         }
//         return x;
//     }

//     private:
//     bool isSoftmax_;
//     torch::nn::Linear linear1_{nullptr};
//     torch::nn::Linear linear2_{nullptr};
//     torch::nn::Linear linear3_{nullptr};
//     torch::nn::Softmax softmax_{nullptr};

// };

// // 定义MLP网络
// struct MLP : torch::nn::Module {
//     torch::nn::Linear layer1{nullptr};
//     torch::nn::Linear layer2{nullptr};
//     torch::nn::Linear layer3{nullptr};

//     MLP() {
//         // 初始化网络层并注册
//         layer1 = register_module("layer1", torch::nn::Linear(torch::nn::LinearOptions(1, 16)));
//         layer2 = register_module("layer2", torch::nn::Linear(torch::nn::LinearOptions(16, 16)));
//         layer3 = register_module("layer3", torch::nn::Linear(torch::nn::LinearOptions(16, 1)));
//     }

//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(layer1(x));
//         x = torch::relu(layer2(x));
//         x = layer3(x);
//         return x;
//     }
// };




// int main() {
//     bool isSoftmax = false;
//     int dimInput = 1;
//     int dimHidden = 16;
//     int dimOutput = 1;
//     std::string optim_type = "SGD";
//     double lr = 0.01;
//     int epochs = 10000;
//     // torch::nn::Sequential model = torch::nn::Sequential({
//     //     {"linear1", torch::nn::Linear(dimInput, dimHidden)},
//     //     {"activation1", torch::nn::Tanh()},
//     //     {"linear2", torch::nn::Linear(dimHidden, dimHidden)},
//     //     {"activation2", torch::nn::Tanh()},
//     //     {"output", torch::nn::Linear(dimHidden, dimOutput)}
//     // });
//     // if (isSoftmax){
//     //     model.push_back("activation3", torch::nn::Softmax(1));  
//     // }



//     auto model = ThreeLayerTanhNN(dimInput, dimHidden, dimOutput, isSoftmax);
//     model.to(device); // 将模型移动到GPU
//     // auto model = MLP();

//     std::cout << "Model Struct:\n" << model << std::endl;

    

//     // 定义优化器和损失函数
//     torch::optim::Adam optimizer_adam(model.parameters(), torch::optim::AdamOptions(0.001));
//     std::unique_ptr<torch::optim::Optimizer> optimizer_uptr;
//     if(optim_type == "SGD"){
//         optimizer_uptr = std::make_unique<torch::optim::SGD>(model.parameters(), /*learning_rate=*/lr);  
//     }else{
//         std::cerr << "Invalid optimizer type: " << optim_type << std::endl;
//         // return;
//     }
//     auto loss_fn = torch::nn::MSELoss();

//     // 生成训练数据
//     auto func = [](float x) { return 2 * std::sin(x) + 4 * std::cos(x) + x*x; };
//     std::vector<float> x_data;
//     std::vector<float> y_data;
//     for (float x = -10; x <= 10; x += 0.1) {
//         x_data.push_back(x);
//         y_data.push_back(func(x));
//     }

//     // 转换为Tensor
//     auto x_tensor = torch::tensor(x_data).unsqueeze(1).to(device);;
//     auto y_tensor = torch::tensor(y_data).unsqueeze(1).to(device);;

//     auto time_start = std::chrono::high_resolution_clock::now();

//     // 训练网络
//     for (int epoch = 0; epoch < epochs; epoch++) {
//         // 前向传播
//         auto y_pred = model.forward(x_tensor);

//         // 计算损失
//         auto loss = loss_fn(y_pred, y_tensor);

//         // 反向传播和优化
//         // optimizer_uptr->zero_grad();
//         optimizer_adam.zero_grad();
//         loss.backward();
//         // optimizer_uptr->step();
//         optimizer_adam.step();

//         // 打印损失
//         if (epoch % (int)(epochs/10) == 0) {//sdf
//             std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
//         }
//     }

//     auto time_end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
//     std::cout << "Training Time: " << duration.count() << " ms" << std::endl;

//     // 打印训练后的损失
//     auto y_pred = model.forward(x_tensor);
//     auto final_loss = loss_fn(y_pred, y_tensor);
//     std::cout << "Final Loss: " << final_loss.item<float>() << std::endl;

//     // 生成测试数据
//     std::vector<float> test_x_data;
//     std::vector<float> test_y_data;
//     for (float x = -10; x <= 10; x += 0.05) {  // 更密集的测试点
//         test_x_data.push_back(x);
//         test_y_data.push_back(func(x));
//     }

//     // 转换为Tensor
//     auto test_x_tensor = torch::tensor(test_x_data).unsqueeze(1).to(device);;

//     // 预测
//     auto test_y_pred = model.forward(test_x_tensor);

//     // 转换为向量
//     std::vector<float> test_y_pred_data(test_y_pred.data_ptr<float>(), test_y_pred.data_ptr<float>() + test_y_pred.numel());

//     // 使用matplotlib-cpp进行可视化
//     plt::figure_size(1200, 800);
//     plt::title("MLP Fitting y = func");
//     plt::xlabel("x");
//     plt::ylabel("y");

//     // 绘制真实值
//     plt::named_plot("True", test_x_data, test_y_data, "b-");

//     // 绘制预测值
//     plt::named_plot("Pred", test_x_data, test_y_pred_data, "r--");

//     // 显示图例
//     plt::legend();

//     // 显示图形
//     plt::show();

//     return 0;
// }




#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <matplotlibcpp.h>  // 包含matplotlib-cpp头文件

namespace plt = matplotlibcpp;  // 使用命名空间

torch::nn::Sequential createSequentialModel(int dimInput, int dimHidden, int dimOutput, bool isSoftmax = true) {
    // Create a sequential model with three layers
    torch::nn::Sequential model = torch::nn::Sequential({
        {"linear1", torch::nn::Linear(dimInput, dimHidden)},
        {"activation1", torch::nn::Tanh()},
        {"linear2", torch::nn::Linear(dimHidden, dimHidden)},
        {"activation2", torch::nn::Tanh()},
        {"output", torch::nn::Linear(dimHidden, dimOutput)}
    });
    if (isSoftmax){
        model->push_back("activation3", torch::nn::Softmax(1));  
    }
    return model;
}

class ThreeLayerTanhNN : public torch::nn::Module {
public:
    ThreeLayerTanhNN(int dimInput, int dimHidden, int dimOutput, bool isSoftmax = true) {
        linear1_ = register_module("linear1", torch::nn::Linear(dimInput, dimHidden));
        tanh1_ = register_module("activation1", torch::nn::Tanh());
        linear2_ = register_module("linear2", torch::nn::Linear(dimHidden, dimHidden));
        tanh2_ = register_module("activation2", torch::nn::Tanh());
        linear3_ = register_module("linear3", torch::nn::Linear(dimHidden, dimOutput));
        isSoftmax_ = isSoftmax;
        if (isSoftmax_) {
            softmax_ = register_module("activation3", torch::nn::Softmax(1));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        x = linear1_->forward(x);
        x = tanh1_->forward(x);
        x = linear2_->forward(x);
        x = tanh2_->forward(x);
        x = linear3_->forward(x);
        if (isSoftmax_) {
            x = softmax_->forward(x);
        }
        return x;
    }

private:
    bool isSoftmax_;
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Tanh tanh1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::Tanh tanh2_{nullptr};
    torch::nn::Linear linear3_{nullptr};
    torch::nn::Softmax softmax_{nullptr};
};

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
    // std::cout << "=====init thread config=====" << std::endl;
    // // 打印当前的intra-op线程数
    // std::cout << "Current intra-op threads: " << torch::get_num_threads() << std::endl;
    // // 打印当前的inter-op线程数
    // std::cout << "Current inter-op threads: " << torch::get_num_interop_threads() << std::endl;

    // // 设置intra-op线程数
    // torch::set_num_threads(4);
    // // 设置inter-op线程数
    // torch::set_num_interop_threads(2);

    // std::cout << "=====after thread config=====" << std::endl;
    // // 打印当前的intra-op线程数
    // std::cout << "Current intra-op threads: " << torch::get_num_threads() << std::endl;
    // // 打印当前的inter-op线程数
    // std::cout << "Current inter-op threads: " << torch::get_num_interop_threads() << std::endl;


    bool isSoftmax = false;
    int dimInput = 1;
    int dimHidden = 8;
    int dimOutput = 1;
    std::string optim_type = "Adam"; // 使用Adam优化器
    double lr = 0.001;
    int epochs = 5000;
    torch::DeviceType device;
    // 检查CUDA是否可用
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Using CPU instead." << std::endl;
        device = torch::kCPU;
        return -1;
    } else {
        device = torch::kCUDA;
        std::cout << "CUDA is available. Using GPU." << std::endl;
    }

    // test
    device = torch::kCUDA; // 强制使用CPU TEST


    std::cout << "device type: " << device << std::endl;

    // 创建模型并移动到GPU
    auto model = ThreeLayerTanhNN(dimInput, dimHidden, dimOutput, isSoftmax);
    model.to(device);

    std::cout << "Model Struct:\n" << model << std::endl;

    // 定义优化器和损失函数
    // torch::optim::Adam optimizer_adam(model.parameters(), torch::optim::AdamOptions(0.001));
    torch::optim::SGD optimizer_sgd(model.parameters(), torch::optim::SGDOptions(lr).momentum(0).dampening(0).weight_decay(0).nesterov(false));
    torch::optim::LBFGS optimizer_lfgs(model.parameters(), torch::optim::LBFGSOptions(lr).max_iter(20).max_eval(30).tolerance_grad(1e-7).tolerance_change(1e-9).history_size(100).line_search_fn("strong_wolfe"));
    torch::optim::Adam optimizer_adam(model.parameters(), torch::optim::AdamOptions(lr));
    // optimizer_adam.to(device)


    auto loss_fn = torch::nn::MSELoss();

    // 生成训练数据
    auto func = [](float x) { return 2 * std::sin(x) + 4 * std::cos(x) + x*x; };
    std::vector<float> x_data;
    std::vector<float> y_data;
    for (float x = -10; x <= 10; x += 0.0001) {
        x_data.push_back(x);
        y_data.push_back(func(x));
    }

    // 转换为Tensor并移动到GPU
    auto x_tensor = torch::tensor(x_data).unsqueeze(1).to(device);
    auto y_tensor = torch::tensor(y_data).unsqueeze(1).to(device);
    std::cout << "=====x_tensor.type()=====" << x_tensor.dtype() << std::endl;

    auto time_start = std::chrono::high_resolution_clock::now();

    // 训练网络
    for (int epoch = 0; epoch < epochs; epoch++) {
        // 前向传播
        auto y_pred = model.forward(x_tensor);

        // 计算损失
        auto loss = loss_fn(y_pred, y_tensor);

        // 反向传播和优化
        optimizer_sgd.zero_grad();
        loss.backward();
        optimizer_sgd.step();

        // 打印损失
        if (epoch % (int)(epochs/10) == 0) {
            std::cout << "Epoch: " << epoch << ", Loss: " << loss.item<float>() << std::endl;
        }
    }

    auto time_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "Training Time: " << duration.count() << " ms" << std::endl;

    // 打印训练后的损失
    auto y_pred = model.forward(x_tensor);
    auto final_loss = loss_fn(y_pred, y_tensor);
    std::cout << "Final Loss: " << final_loss.item<float>() << std::endl;

    // 生成测试数据
    std::vector<float> test_x_data;
    std::vector<float> test_y_data;
    for (float x = -10; x <= 10; x += 0.05) {  // 更密集的测试点
        test_x_data.push_back(x);
        test_y_data.push_back(func(x));
    }

    // 转换为Tensor并移动到GPU
    auto test_x_tensor = torch::tensor(test_x_data).unsqueeze(1).to(device);

    // 预测
    auto test_y_pred = model.forward(test_x_tensor);

    // 将预测结果移动回CPU并转换为向量
    auto test_y_pred_cpu = test_y_pred.to(torch::kCPU);
    std::vector<float> test_y_pred_data(test_y_pred_cpu.data_ptr<float>(), test_y_pred_cpu.data_ptr<float>() + test_y_pred_cpu.numel());

    // 使用matplotlib-cpp进行可视化
    plt::figure_size(1200, 800);
    plt::title("MLP Fitting y = func");
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