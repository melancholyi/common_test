#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include <ATen/ATen.h>

// 真实高程函数
torch::Tensor true_elevation(const torch::Tensor& X) {
    auto x = X.index({torch::indexing::Slice(), 0});
    auto y = X.index({torch::indexing::Slice(), 1});
    return 10 * torch::sin(5 * x) + 10 * torch::cos(5 * y) + 10 * x + y;
}

// SGP模型类
class SparseGaussianProcess : public torch::nn::Module {
public:
    SparseGaussianProcess(torch::Tensor inducing_points, bool learn_inducing = true)
        : M(inducing_points.size(0)), D(inducing_points.size(1)) {
        // 初始化诱导点
        register_parameter("inducing_points", 
            torch::nn::Parameter(inducing_points.clone(), learn_inducing));
        
        // 变分分布参数
        register_parameter("variational_mean",
            torch::nn::Parameter(torch::zeros(M)));
        register_parameter("chol_variational_covar",
            torch::nn::Parameter(torch::eye(M)));
        
        // 均值模块
        register_parameter("mean_constant", torch::nn::Parameter(torch::zeros(1)));
        
        // RBF核参数
        register_parameter("log_lengthscale", 
            torch::nn::Parameter(torch::zeros(1)));
        register_parameter("log_outputscale",
            torch::nn::Parameter(torch::zeros(1)));
        
        // 噪声参数
        register_parameter("log_noise", torch::nn::Parameter(torch::zeros(1)));
    }

    // 核函数计算
    torch::Tensor kernel(torch::Tensor X, torch::Tensor Y = torch::Tensor()) {
        if (!Y.defined()) Y = X;
        auto scale = torch::exp(log_lengthscale);
        X = X / scale;
        Y = Y / scale;
        auto dist = torch::cdist(X, Y);
        return torch::exp(2 * log_outputscale) * torch::exp(-0.5 * dist.pow(2));
    }

    // 训练前向传播
    std::tuple<torch::Tensor, torch::Tensor> forward_train(torch::Tensor X, torch::Tensor y) {
        auto inducing = inducing_points;
        auto K_uu = kernel(inducing) + torch::eye(M, torch::kDouble) * 1e-6;
        auto K_uu_inv = torch::inverse(K_uu);
        
        // 变分参数
        auto L = torch::tril(chol_variational_covar);
        auto S = L.mm(L.t());
        auto m_u = variational_mean;
        auto prior_mu = mean_constant.expand({M});
        
        // KL散度
        auto prior_dist = torch::distributions::MultivariateNormal(
            prior_mu, torch::linalg::cholesky(K_uu));
        auto var_dist = torch::distributions::MultivariateNormal(
            m_u, torch::linalg::cholesky(S));
        auto kl = torch::distributions::kl_divergence(var_dist, prior_dist);
        
        // 预测分布参数
        auto K_fu = kernel(X, inducing);
        auto mean_f = mean_constant + K_fu.mm(K_uu_inv).mv(m_u - prior_mu);
        
        // 计算对数似然
        auto noise = torch::exp(log_noise);
        auto y_diff = y - mean_f;
        auto term1 = y_diff.dot(y_diff);
        
        // 迹计算
        auto trace_Kff = X.size(0) * torch::exp(2 * log_outputscale);
        auto q_fu = K_fu.mm(K_uu_inv);
        auto trace_term2 = (q_fu * K_fu).sum();
        auto A = K_uu_inv.mm(S).mm(K_uu_inv);
        auto trace_term3 = (K_fu.mm(A) * K_fu).sum();
        auto trace_total = trace_Kff - trace_term2 + trace_term3 + X.size(0) * noise;
        
        // 对数似然
        auto log_likelihood = -0.5 * X.size(0) * (torch::log(2 * M_PI * noise) + 
                         (term1 + trace_total) / noise);
        
        return {log_likelihood, kl};
    }

    // 预测
    std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor X_test) {
        auto inducing = inducing_points;
        auto K_uu = kernel(inducing) + torch::eye(M, torch::kDouble) * 1e-6;
        auto K_uu_inv = torch::inverse(K_uu);
        auto K_su = kernel(X_test, inducing);
        
        // 均值预测
        auto prior_mu = mean_constant.expand({M});
        auto mean_adj = variational_mean - prior_mu;
        auto mean = mean_constant + K_su.mm(K_uu_inv).mv(mean_adj);
        
        // 方差预测
        auto q_su = K_su.mm(K_uu_inv);
        auto term1 = (q_su * K_su).sum(1);
        auto L = torch::tril(chol_variational_covar);
        auto S = L.mm(L.t());
        auto A = K_uu_inv.mm(S).mm(K_uu_inv);
        auto term2 = (K_su.mm(A) * K_su).sum(1);
        auto noise = torch::exp(log_noise);
        auto var = torch::exp(2 * log_outputscale) - term1 + term2 + noise;
        
        return {mean, var};
    }

private:
    int64_t M, D;  // 诱导点数量和输入维度
};

// 测试代码
int main() {
    // 生成数据
    double min_val = 0, max_val = 10, res = 0.2;
    int64_t D = 2;
    int64_t Ngrid = (max_val - min_val)/res + 1;
    
    auto grid = torch::meshgrid({
        torch::linspace(min_val, max_val, Ngrid),
        torch::linspace(min_val, max_val, Ngrid)
    });
    auto X_train = torch::stack({grid[0].flatten(), grid[1].flatten()}, -1);
    auto y_train = true_elevation(X_train) + torch::randn(X_train.size(0)) * 0.01;
    
    // 选择诱导点
    auto inducing_points = X_train.index({torch::arange(0, X_train.size(0), 100});
    
    // 初始化模型
    auto model = std::make_shared<SparseGaussianProcess>(inducing_points, true);
    
    // 优化器
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.1));
    
    // 训练循环
    int epochs = 500;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        optimizer.zero_grad();
        auto [log_likelihood, kl] = model->forward_train(X_train, y_train);
        auto loss = -(log_likelihood - kl);
        loss.backward();
        optimizer.step();
        std::cout << "Epoch " << epoch+1 << "/" << epochs 
                  << " - Loss: " << loss.item() << std::endl;
    }
    
    // 预测
    model->eval();
    auto [y_pred, var_pred] = model->predict(X_train);
    auto mse = torch::mean(torch::square(y_pred - true_elevation(X_train)));
    std::cout << "MSE: " << mse.item() << std::endl;

    return 0;
}