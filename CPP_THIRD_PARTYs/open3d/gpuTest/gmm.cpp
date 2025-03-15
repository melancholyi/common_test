// #include <iostream>
// #include <open3d/core/Tensor.h>
// #include <open3d/core/Device.h>
// #include <open3d/core/Dispatch.h>
// #include <open3d/core/Kernel.h>
// #include <open3d/core/Linalg.h>
// #include <open3d/core/Random.h>
// #include <open3d/core/Utils.h>

// using namespace open3d::core;

// // 计算后验概率（E-step）
// Tensor ComputeResponsibilities(const Tensor& data, const Tensor& means, const Tensor& covariances, const Tensor& weights) {
//     auto device = data.GetDevice();
//     int num_samples = data.GetShape()[0];
//     int num_components = means.GetShape()[0];
//     int num_features = data.GetShape()[1];

//     Tensor responsibilities = Tensor::Zeros({num_samples, num_components}, Dtype::Float32, device);

//     for (int k = 0; k < num_components; ++k) {
//         Tensor diff = data - means.Slice({k}, {k + 1}).BroadcastTo({num_samples, num_features});
//         Tensor inv_cov = Tensor::Inverse(covariances.Slice({k}, {k + 1}).Squeeze(0));
//         Tensor det_cov = Tensor::Det(covariances.Slice({k}, {k + 1}).Squeeze(0));
//         Tensor exponent = -0.5 * Tensor::Sum(diff.MatMul(inv_cov).MatMul(diff.T()), -1);
//         Tensor pdf = weights[k] * Tensor::Exp(exponent) / Tensor::Sqrt(det_cov.Pow(num_features));

//         responsibilities.Slice({0}, {num_samples}, {k}, {k + 1}) += pdf.Reshape({num_samples, 1});
//     }

//     responsibilities = responsibilities / Tensor::Sum(responsibilities, 1, true);
//     return responsibilities;
// }

// // 更新均值（M-step）
// Tensor UpdateMeans(const Tensor& data, const Tensor& responsibilities) {
//     int num_components = responsibilities.GetShape()[1];
//     int num_features = data.GetShape()[1];

//     Tensor new_means = Tensor::Zeros({num_components, num_features}, Dtype::Float32, data.GetDevice());

//     for (int k = 0; k < num_components; ++k) {
//         Tensor resp_k = responsibilities.Slice({0}, {-1}, {k}, {k + 1}).Squeeze(1);
//         new_means.Slice({k}, {k + 1}) = (resp_k * data).Sum(0, true) / resp_k.Sum();
//     }

//     return new_means;
// }

// // 更新协方差矩阵（M-step）
// Tensor UpdateCovariances(const Tensor& data, const Tensor& responsibilities, const Tensor& means) {
//     int num_components = responsibilities.GetShape()[1];
//     int num_features = data.GetShape()[1];

//     Tensor new_covariances = Tensor::Zeros({num_components, num_features, num_features}, Dtype::Float32, data.GetDevice());

//     for (int k = 0; k < num_components; ++k) {
//         Tensor resp_k = responsibilities.Slice({0}, {-1}, {k}, {k + 1}).Squeeze(1);
//         Tensor mean_k = means.Slice({k}, {k + 1}).Squeeze(0);
//         Tensor diff = data - mean_k.BroadcastTo({data.GetShape()[0], num_features});
//         new_covariances.Slice({k}, {k + 1}) = (resp_k * diff.MatMul(diff.T())).Sum(0, true) / resp_k.Sum();
//     }

//     return new_covariances;
// }

// // 更新混合权重（M-step）
// Tensor UpdateWeights(const Tensor& responsibilities) {
//     return Tensor::Sum(responsibilities, 0) / responsibilities.GetShape()[0];
// }

// // 检查收敛性
// bool CheckConvergence(const Tensor& old_means, const Tensor& new_means,
//                       const Tensor& old_covariances, const Tensor& new_covariances,
//                       const Tensor& old_weights, const Tensor& new_weights,
//                       float tol) {
//     float mean_diff = (new_means - old_means).Abs().Max().Item<float>();
//     float cov_diff = (new_covariances - old_covariances).Abs().Max().Item<float>();
//     float weight_diff = (new_weights - old_weights).Abs().Max().Item<float>();

//     return mean_diff < tol && cov_diff < tol && weight_diff < tol;
// }

// // GMM算法主函数
// void GMM(Tensor& data, int num_components, int max_iter, float tol) {
//     auto device = data.GetDevice();
//     int num_samples = data.GetShape()[0];
//     int num_features = data.GetShape()[1];

//     Tensor means = Tensor::Random<float>({num_components, num_features}).To(device);
//     Tensor covariances = Tensor::Eye(num_features).To(device).BroadcastTo({num_components, num_features, num_features});
//     Tensor weights = Tensor::Full<float>({num_components}, 1.0 / num_components).To(device);

//     for (int iter = 0; iter < max_iter; ++iter) {
//         Tensor responsibilities = ComputeResponsibilities(data, means, covariances, weights);

//         Tensor old_means = means;
//         Tensor old_covariances = covariances;
//         Tensor old_weights = weights;

//         means = UpdateMeans(data, responsibilities);
//         covariances = UpdateCovariances(data, responsibilities, means);
//         weights = UpdateWeights(responsibilities);

//         if (CheckConvergence(old_means, means, old_covariances, covariances, old_weights, weights, tol)) {
//             std::cout << "GMM converged at iteration " << iter << std::endl;
//             break;
//         }
//     }
// }

// // 主函数：生成随机数据并运行GMM
// int main() {
//     // 设置设备为GPU
//     Device device("CUDA:0");

//     // 生成随机数据（2D数据，3个高斯分布）
//     Tensor data = Tensor::Random<float>({300, 2}, device);
//     data.Slice({0}, {100}) += Tensor::Full<float>({100, 2}, {0.0, 0.0}).To(device);
//     data.Slice({100}, {200}) += Tensor::Full<float>({100, 2}, {5.0, 5.0}).To(device);
//     data.Slice({200}, {300}) += Tensor::Full<float>({100, 2}, {-5.0, -5.0}).To(device);

//     // 运行GMM算法
//     int num_components = 3;
//     int max_iter = 100;
//     float tol = 1e-4;

//     GMM(data, num_components, max_iter, tol);

//     std::cout << "GMM clustering completed." << std::endl;

//     return 0;
// }