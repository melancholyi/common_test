#include "torch/torch.h"
#include "matplotlibcpp.h"


#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>


//====================PART:1  createSequentialModel ========================
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

    // Print the model structure
    std::cout << "=====PART1 createSequentialModel:" << std::endl;
    std::cout << "1. Model structure: " << model << std::endl;

    // use input and output to test the model
    torch::Tensor input = torch::ones({1, dimInput});  // allOnes input with shape [1, dimInput]
    torch::Tensor output = model->forward(input);  // Forward pass
    std::cout << "2. Output shape: " << output.sizes() << std::endl;
    std::cout << "3. Output values: " << output << std::endl;
    return model;
}

//====================PART:2  createOptimizerTofittingFunc ========================

class ThreeLayerTanhNN : public torch::nn::Sequential {
public:
    ThreeLayerTanhNN(int dimInput, int dimHidden, int dimOutput, bool isSoftmax = true) : Sequential(torch::nn::Sequential(
        {{"linear1", torch::nn::Linear(dimInput, dimHidden)},
        {"activation1", torch::nn::Tanh()},
        {"linear2", torch::nn::Linear(dimHidden, dimHidden)},
        {"activation2", torch::nn::Tanh()},
        {"output", torch::nn::Linear(dimHidden, dimOutput)},
        {"activation3", torch::nn::Softmax(1)}}
    )) {
        
    }
};

struct MLP : torch::nn::Module {
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    torch::nn::Linear layer3{nullptr};

    MLP() {
        // 初始化网络层并注册
        layer1 = register_module("layer1", torch::nn::Linear(1, 16));
        layer2 = register_module("layer2", torch::nn::Linear(16, 16));
        layer3 = register_module("layer3", torch::nn::Linear(16, 1));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1(x));
        x = torch::relu(layer2(x));
        x = layer3(x);
        return x;
    }
};



void createOptimizerTofitFunc() {

    //STEP:1 ========== parameters ================
    double noise_std = 1; // noise standard deviation

    //optimizer parameters
    std::string optim_type = "SGD"; // optimizer type

    int epochs = 1000; // max epoch
    double lr = 0.01; // learning rate
    int max_iter = 1000; // max iteration
    double tolerance = 1e-4; // tolerance
    
    
    //STEP:2 ========== create dataset ================
    namespace plt = matplotlibcpp;

    // create dataset
    // 生成 x 值（在 [-10, 10] 范围内均匀分布）
    int num_samples = 1000;
    int num_points = 100;
    std::vector<double> x(num_samples);
    std::vector<double> xpt(num_points);
    for (int i = 0; i < num_samples; ++i) {
        x[i] = -10 + (20.0 * i) / (num_samples - 1);
    }
    for (int i = 0; i < num_points; ++i) {
        xpt[i] = -10 + (20.0 * i) / (num_points - 1);
    }

    // 生成 y 值：y = 2*sin(x) + 3*cos(x) + x^2
    std::vector<double> y(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        double x_val = x[i];
        y[i] = 2 * std::sin(x_val) + 3 * std::cos(x_val) + x_val * x_val;
    }

    // 添加随机噪声
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, noise_std); 

    std::vector<double> y_noisy(num_points);
    for (int i = 0; i < num_points; ++i) {
        auto x_val = xpt[i];
        auto temp = 2 * std::sin(x_val) + 3 * std::cos(x_val) + x_val * x_val;
        y_noisy[i] = temp + noise_dist(gen);
    }

    //STEP:3 =========== convert to tensor ==============
    // Convert data to torch tensors
    torch::Tensor x_tensor = torch::tensor(xpt, torch::kFloat32).view({-1, 1});
    torch::Tensor y_tensor = torch::tensor(y_noisy, torch::kFloat32).view({-1, 1});
    
    //STEP:4 =========== model fitting func ==============
    
    // auto model = createSequentialModel(1, 8, 1);
    // auto model = ThreeLayerTanhNN(1, 8, 1);
    auto model = std::make_shared<MLP>();

    std::cout << "\n=====PART2 createOptimizerTofittingFunc:" << std::endl;
    // Create a loss function
    auto loss_function = torch::nn::MSELoss();
    // Create an optimizer

    
    std::unique_ptr<torch::optim::Optimizer> optimizer_uptr;
    if(optim_type == "SGD"){
        optimizer_uptr = std::make_unique<torch::optim::SGD>(model->parameters(), /*learning_rate=*/lr);  
    }else{
        std::cerr << "Invalid optimizer type: " << optim_type << std::endl;
        return;
    }

    torch::optim::SGD optimizer_sgd(model->parameters(), /*learning_rate=*/lr);


    for(int i = 0 ;i < epochs; i++){
        // Zero gradients
        // optimizer_uptr->zero_grad();
        optimizer_sgd.zero_grad();

        // Forward pass
        torch::Tensor output = model->forward(x_tensor);

        // Compute loss
        torch::Tensor loss = loss_function(output, y_tensor);

        // std::cout << "dsebug: " << "x_tensor.size: " << x_tensor.sizes() << std::endl;
        // std::cout << "dsebug: " << "y_tensor.size: " << y_tensor.sizes() << std::endl;
        // std::cout << "dsebug: " << "output.size: " << output.sizes() << std::endl;
        // std::cout << "dsebug: " << "loss.size: " << loss.sizes() << std::endl;
        std::cout << "Training Epoch [" << i + 1 << "/" << epochs << "], Loss: " << loss.item<double>() << std::endl;

        // Backward pass
        loss.backward();

        // Update weights
        // optimizer_uptr->step();
        optimizer_sgd.step();
    }



    // Print
    
    if(optim_type == "SGD"){
        std::cout << "1. Optimizer structure: " << "torch::optim::SGD" << std::endl;
    }
    std::cout << "2. Loss function structure: " << loss_function << std::endl;
    // Print the model structure
    std::cout << "3. Model structure: " << model << std::endl;
    
    



    //STEP:3 =========== plot the function with noise ==============
    // 使用 matplotlib-cpp 绘制图像
    plt::figure_size(1200, 780);

    // 绘制原始函数
    plt::named_plot("yTrue = 2*sin(x) + 3*cos(x) + x²", x, y, "r-");

    // 绘制带噪声的训练数据
    plt::scatter(xpt, y_noisy, 50);

    plt::title("Function Visualization with Noise");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::legend();
    plt::grid(true);
    plt::show();


    // // Create a model
    // auto model = createSequentialModel(1, 128, 1, true);

    // // Create a loss function
    // auto loss_function = torch::nn::CrossEntropyLoss();

    // // Create an optimizer
    // torch::optim::SGD optimizer(model->parameters(), /*learning_rate=*/0.01);

    // // Print the optimizer structure
    // std::cout << "=====PART2 createOptimizerTofittingFunc:" << std::endl;
    // std::cout << "1. Optimizer structure: " << optimizer << std::endl;
}



int main(){
    //====================PART:1  createSequentialModel ========================
    auto seq_model1 = createSequentialModel(784, 128, 10, true);
    
    //====================PART:2  createOptimizerTofittingFunc ========================
    createOptimizerTofitFunc();





    return 0;
}