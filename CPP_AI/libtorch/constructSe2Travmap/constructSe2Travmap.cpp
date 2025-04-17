/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-16 07:40:36
 * @LastEditTime: 2025-04-17 13:41:28
 * @FilePath: /test/CPP_AI/libtorch/constructSe2Travmap/constructSe2Travmap.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <torch/torch.h>

int main() {
    //PART:1 create data ////////////////////////////////////////////////////////////////////////////////////////////
    // 创建一个包含25个 Eigen::Vector3d 的 std::vector
    std::vector<Eigen::Vector3d> eigen_vectors(25);
    for (int i = 0; i < 25; ++i) {
        eigen_vectors[i] = Eigen::Vector3d(i * 1, i * 2, i * 3);
    }
    // 打印 Eigen::Vector3d 的内容
    std::cout << "eigen_vectors simulation 5x5 mean3D data:" << std::endl;
    for(int i = 0; i < 25; ++i) {
        // 使用 std::setw 和 std::setfill 确保每个数字是两位数
        std::cout << "(" 
            << std::setw(2) << std::setfill('0') << eigen_vectors[i](0) << ", " 
            << std::setw(2) << std::setfill('0') << eigen_vectors[i](1) << ", " 
            << std::setw(2) << std::setfill('0') << eigen_vectors[i](2) 
            << ") ";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::setw(0) << std::setfill(' ');
    /*
    (00, 00, 00) (01, 02, 03) (02, 04, 06) (03, 06, 09) (04, 08, 12)
    (05, 10, 15) (06, 12, 18) (07, 14, 21) (08, 16, 24) (09, 18, 27)
    (10, 20, 30) (11, 22, 33) (12, 24, 36) (13, 26, 39) (14, 28, 42)
    (15, 30, 45) (16, 32, 48) (17, 34, 51) (18, 36, 54) (19, 38, 57)
    (20, 40, 60) (21, 42, 63) (22, 44, 66) (23, 46, 69) (24, 48, 72)
    */

    // 检查是否有可用的 CUDA 设备
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Falling back to CPU." << std::endl;
        return 1;
    }

    // 使用 torch::from_blob 创建 Tensor，避免数据复制
    // Eigen::Vector3d 是连续存储的，因此可以直接使用 data() 获取指针
    torch::Tensor tensor = torch::from_blob(
        eigen_vectors.data(),  // 数据指针
        {25, 3},               // 数据形状 (25x3)
        torch::kDouble         // 数据类型
    ).to(torch::kCUDA);        // 移动到 CUDA 设备

    // 调整 Tensor 的形状为 5x5x3
    tensor = tensor.view({5, 5, 3});

    // 打印 Tensor 的形状
    std::cout << "Tensor shape: " << tensor.sizes() << std::endl;

    // 将 Tensor 移回 CPU 以便打印
    torch::Tensor tensor_cpu = tensor.to(torch::kCPU);
    std::cout << "tensor_cpu.size()" << tensor_cpu.sizes() << std::endl;

    //PART: 2 printf tensor data /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 打印 Tensor 的部分数据
    // std::cout << tensor_cpu.index({torch::indexing::Slice(0, 2), torch::indexing::Slice(0, 2), torch::indexing::Slice()}) << std::endl;
    std::cout << "mean.x first channel" << std::endl;
    // 提取第一个通道，形状为 [5, 5]
    torch::Tensor first_channel = tensor_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}).unsqueeze(2).view({5, 5});
    std::cout << first_channel << std::endl;
    // 提取第2个通道，形状为 [5, 5]
    torch::Tensor second_channel = tensor_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(), 1}).unsqueeze(2).view({5, 5});
    std::cout << second_channel << std::endl;
    // 提取第3个通道，形状为 [5, 5]
    torch::Tensor third_channel = tensor_cpu.index({torch::indexing::Slice(), torch::indexing::Slice(), 2}).unsqueeze(2).view({5, 5});
    std::cout << third_channel << std::endl;

    // 打印all
    std::cout << "print all Eigen::Vector3d: " << std::endl;
    for(int i = 0 ; i < 5; i++){
        for(int j = 0; j < 5; j++){
            std::cout << "(" << i << "," << j << "):" << tensor_cpu[i][j].view({3,1}).transpose(0, 1) << std::endl;
        }
        std::cout << std::endl;
    }


    //PART:3 extract ellipsoid data ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "\n=====PART:3 extract ellipsoid data" << std::endl;
    std::cout << "=====PART:3.1 tensor use slipping windows" << std::endl;
    // 创建一个5x5的随机张量并移动到GPU
    torch::Tensor original_tensor = torch::arange(1.0, 26.0).reshape({5, 5});
    // 使用unfold提取3x3的滑动窗口 
    torch::Tensor unfolded = original_tensor.unfold(0, 3, 1).unfold(1, 3, 1); //NOTE: unfold(dim, window_size, step)
    std::cout << "original_tensor.unfold(0, 3, 1).sizes():" << original_tensor.unfold(0, 3, 1).sizes() << std::endl;
    std::cout << "original_tensor.unfold(0, 3, 1).unfold(1, 3, 1):" << original_tensor.unfold(0, 3, 1).unfold(1, 3, 1).sizes() << std::endl;
    std::cout << "unfolded tensor" << unfolded << std::endl; //size: 3 3 3 3
    std::cout << "print first coviance tensor" << std::endl;
    auto first_tensor = unfolded[0][0];
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            std::cout << "[" << i << "][" << j << "]:" << (float)((first_tensor[i][j]).item<float>()) << " ";
        }
        std::cout << std::endl;
    }
    /**
     * original_tensor: 
     * size{5, 5}
     *  1  2  3  4  5
     *  6  7  8  9 10
     *  11 12 13 14 15
     *  16 17 18 19 20
     *  21 22 23 24 25
     * 
     * original_tensor.unfold(0, 3, 1) 表示沿着dim = 0维度展开，窗口大小为3，步长为1 | dim=0:按行操作，对每一列进行操作。
     * size{3, 5, 3} // 3(first): 表示展开后的维度，5:size(dim = 1) ，3:window_size
     * (1, ..., ...)
     * 1  2  3  4  5
     * 6  7  8  9  10 
     * 11 12 13 14 15
     * 
     * (2, ..., ...)
     * 6  7  8  9 10
     * 11 12 13 14 15
     * 16 17 18 19 20
     * 
     * (3, ..., ...)
     * 11 12 13 14 15
     * 16 17 18 19 20
     * 21 22 23 24 25
     * 
     * original_tensor.unfold(0, 3, 1).unfold(1, 3, 1):[3, 3, 3, 3]
     * 在上面基础上{3, 5, 3} 沿着dim = 1展开， 窗口大小为3，步长为1 | dim=1:按列操作，对每一行进行操作。 
     * 其实就是保留第一维度。然后后续的 {5,3}展开到{3,3,3}. 3(first:) 5->3 3(first: origin dim) 3(third):windows_size
     *   unfolded tensor
     *   (1,1,.,.) =    (1,2,.,.) =   (1,3,.,.) = 
     *   1   2   3      2   3   4     3   4   5   
     *   6   7   8      7   8   9     8   9  10   
     *   11  12  13     12  13  14    13  14  15    
     *
     *   (2,1,.,.) =    (2,2,.,.) =   (2,3,.,.) =
     *   6   7   8      7   8   9     8   9  10            
     *   11  12  13     12  13  14    13  14  15        
     *   16  17  18     17  18  19    18  19  20   
     * 
     *   (3,1,.,.) =    (3,2,.,.) =   (3,3,.,.) =          
     *   11  12  13     12  13  14    13  14  15          
     *   16  17  18     17  18  19    18  19  20          
     *   21  22  23     22  23  24    23  24  25         
     *   [ CPUFloatType{3,3,3,3} ]
     */
    torch::Tensor contiguous_unfolded = unfolded.contiguous();
    torch::Tensor new_tensor = contiguous_unfolded.view({3, 3, 9});
    new_tensor = new_tensor.cpu();
    std::cout << "Original Tensor 5x5:\n" << original_tensor.cpu() << std::endl;
    std::cout << "New Tensor (3x3x9):" << new_tensor << std::endl;


    std::cout << "\n=====PART:3.2 tensor (5x5) -> (3,3,9)" << std::endl;
    torch::Tensor center_patch = original_tensor.index({torch::indexing::Slice(1, 4), torch::indexing::Slice(1, 4)});
    std::cout << "center_patch(3x3): \n" << center_patch << std::endl;
    torch::Tensor expanded_tensor = center_patch.unsqueeze(2).expand({3, 3, 9});
    std::cout << "Expanded Tensor (3x3x9):\n" << expanded_tensor << std::endl;

    std::cout << "\n=====PART:3.3 extrack ellipsoid mask" << std::endl;
    //private namespace
    {   
        //! parameters
        double resolution = 0.2; // meters
        double robot_L = 1.0, robot_W = 0.7;
        int yaw_divide_num = 12; // 30 degree

        //! NOTE: get the ellipsoid mask tensor shape
        double ellipsoid_a = std::sqrt((robot_L * robot_L + robot_W * robot_W) / 4.0); // semi-major axis 0.610328
        double ellipsoid_b = robot_W/2; // semi-minor axis
        int grid_count = std::round(ellipsoid_a / resolution); // 3
        int exact_grid_len = grid_count * 2 + 1; 


        std::cout << "ellipsoid_a:" << ellipsoid_a << std::endl; //ellipsoid_a:0.610328
        std::cout << "ellipsoid_b:" << ellipsoid_b << std::endl; //ellipsoid_b:0.35
        std::cout << "grid_count:" << grid_count << std::endl; //grid_count:3
        std::cout << "exact_grid_len:" << exact_grid_len << std::endl; //exact_grid_len:7

        //! 10 x 10 origin data 
        torch::Tensor elev_tensor = torch::arange(1.0, 101.0).reshape({10, 10});
        std::cout << "elev_tensor size:" << elev_tensor.sizes() << std::endl;
        std::cout << "elev_tensor:\n" << elev_tensor << std::endl;

        //! unfold extract around data 
        //- elev_tensor.sizes() - grid_count * 2
        //- dim0: 10 - 3 * 2 = 4
        //- dim1: 10 - 3 * 2 = 4
        torch::Tensor elev_tensor_unfold = elev_tensor.unfold(0, exact_grid_len, 1).unfold(1, exact_grid_len, 1);
        std::cout << "elev_tensor_unfold size:" << elev_tensor_unfold.sizes() << std::endl;//[4, 4, 7, 7]
        std::cout << "elev_tensor_unfold:\n" << elev_tensor_unfold << std::endl;
        /*
        origin data:
         1    2    3    4    5    6    7    8    9   10
        11   12   13   14   15   16   17   18   19   20
        21   22   23   24   25   26   27   28   29   30
        31   32   33   34   35   36   37   38   39   40
        41   42   43   44   45   46   47   48   49   50
        51   52   53   54   55   56   57   58   59   60
        61   62   63   64   65   66   67   68   69   70
        71   72   73   74   75   76   77   78   79   80
        81   82   83   84   85   86   87   88   89   90
        91   92   93   94   95   96   97   98   99  100   
        
        (1,1,.,.) =                   (1,2,.,.) =                    ....
        1   2   3   4   5   6   7      2   3   4   5   6   7   8      
        11  12  13  14  15  16  17    12  13  14  15  16  17  18    
        21  22  23  24  25  26  27    22  23  24  25  26  27  28    
        31  32  33  34  35  36  37    32  33  34  35  36  37  38    
        41  42  43  44  45  46  47    42  43  44  45  46  47  48    
        51  52  53  54  55  56  57    52  53  54  55  56  57  58    
        61  62  63  64  65  66  67    62  63  64  65  66  67  68

        (2,1,.,.) =                   (2,2,.,.) =                                  
        11  12  13  14  15  16  17    12  13  14  15  16  17  18          
        21  22  23  24  25  26  27    22  23  24  25  26  27  28          
        31  32  33  34  35  36  37    32  33  34  35  36  37  38          
        41  42  43  44  45  46  47    42  43  44  45  46  47  48          
        51  52  53  54  55  56  57    52  53  54  55  56  57  58          
        61  62  63  64  65  66  67    62  63  64  65  66  67  68          
        71  72  73  74  75  76  77    72  73  74  75  76  77  78          

        (3,1,.,.) =                   (3,2,.,.) =    
        21  22  23  24  25  26  27    22  23  24  25  26  27  28 
        31  32  33  34  35  36  37    32  33  34  35  36  37  38 
        41  42  43  44  45  46  47    42  43  44  45  46  47  48 
        51  52  53  54  55  56  57    52  53  54  55  56  57  58 
        61  62  63  64  65  66  67    62  63  64  65  66  67  68 
        71  72  73  74  75  76  77    72  73  74  75  76  77  78 
        81  82  83  84  85  86  87    82  83  84  85  86  87  88 

        (4,1,.,.) =                    (4,2,.,.) =     
        31  32  33  34  35  36  37     32  33  34  35  36  37  38  
        41  42  43  44  45  46  47     42  43  44  45  46  47  48
        51  52  53  54  55  56  57     52  53  54  55  56  57  58
        61  62  63  64  65  66  67     62  63  64  65  66  67  68
        71  72  73  74  75  76  77     72  73  74  75  76  77  78
        81  82  83  84  85  86  87     82  83  84  85  86  87  88
        91  92  93  94  95  96  97     92  93  94  95  96  97  98                                                                                               
        */

        //! ellipsoid mask 
        //theta test 0 30 60 90 120 150 180 210 240 270 300 330 (12 num data)
        
        torch::Tensor angles = torch::linspace(0, 360 - 360/yaw_divide_num, yaw_divide_num);
        std::cout << "angles:" << angles << std::endl;
        std::cout << "torch::Tensor x = torch::arange(-grid_count, grid_count + 1).to(torch::kDouble):\n" << torch::arange(-grid_count, grid_count + 1).to(torch::kDouble) << std::endl;
        std::vector<torch::Tensor> angles_masks;
        angles_masks.reserve(yaw_divide_num);
        for (int i = 0; i < yaw_divide_num; ++i) {
            double angle = angles[i].item<double>() * M_PI / 180.0;
            std::cout << "angle:" << angle << std::endl;

            // 生成 x 和 y 的网格
            torch::Tensor x = torch::arange(-grid_count, grid_count + 1).to(torch::kDouble); // -3 -2 -1 0 1 2 3
            torch::Tensor y = torch::arange(-grid_count, grid_count + 1).to(torch::kDouble);

            // 创建网格点
            torch::Tensor X = x.unsqueeze(0).repeat({y.size(0), 1}); // X: [7, 7]
            torch::Tensor Y = y.unsqueeze(1).repeat({1, x.size(0)}); // Y: [7, 7]
            /*
            Y:                      X:  
            -3 -3 -3 -3 -3 -3 -3    -3 -2 -1  0  1  2  3   
            -2 -2 -2 -2 -2 -2 -2    -3 -2 -1  0  1  2  3  
            -1 -1 -1 -1 -1 -1 -1    -3 -2 -1  0  1  2  3    
             0  0  0  0  0  0  0    -3 -2 -1  0  1  2  3    
             1  1  1  1  1  1  1    -3 -2 -1  0  1  2  3   
             2  2  2  2  2  2  2    -3 -2 -1  0  1  2  3    
             3  3  3  3  3  3  3    -3 -2 -1  0  1  2  3   
            */
            // std::cout << "X size:" << X.sizes() << std::endl;
            // std::cout << "Y size:" << Y.sizes() << std::endl;
            // std::cout << "X:\n" << X << std::endl;
            // std::cout << "Y:\n" << Y << std::endl;

            // 计算旋转后的坐标
            torch::Tensor rotated_x = X * std::cos(-angle) - Y * std::sin(-angle);
            torch::Tensor rotated_y = X * std::sin(-angle) + Y * std::cos(angle);
            rotated_x *= resolution; // 乘以分辨率
            rotated_y *= resolution;

            // 判断是否在椭圆内
            torch::Tensor mask = (rotated_x * rotated_x) / (ellipsoid_a * ellipsoid_a) + 
                                (rotated_y * rotated_y) / (ellipsoid_b * ellipsoid_b) <= 1.0;
            mask = mask.unsqueeze(0).unsqueeze(0).expand({elev_tensor_unfold.size(0), elev_tensor_unfold.size(1), exact_grid_len, exact_grid_len});
            std::cout << "yaw_angle:" << angle << " yaw_angle_degree:" << angles[i].item<double>() << std::endl;
            std::cout << "mask size:" << mask.sizes() << std::endl;
            // std::cout << "mask:\n" << mask << std::endl;////test OK
            auto extract_data = elev_tensor_unfold * mask;
            // std::cout << "extract_data:\n" << extract_data << std::endl;//test OK
            angles_masks.push_back(mask);
            std::cout << "angles_masks[" << i << "].sizes()" << angles_masks[i].sizes() << std::endl;
        }
    }
    



    // std::cout << "\n=====PART:3.4 merge gaussians" << std::endl;
    // torch::Tensor data1 = torch::arange(1.0, 16.0).view({5, 3});
    // torch::Tensor data2 = torch::arange(17.0, 29.0).view({4, 3});
    // // 打印原始数据
    // std::cout << "data1:\n" << data1 << std::endl;
    // std::cout << "data2:\n" << data2 << std::endl;

    // // 分别计算均值和方差
    // auto mean1 = data1.mean(/*dim=*/0);
    // auto var1 = torch::cov(data1.transpose(0, 1), /*correction=*/0);

    // auto mean2 = data2.mean(/*dim=*/0);
    // auto var2 = torch::cov(data2.transpose(0, 1), /*correction=*/0);

    // // 打印均值和方差
    // std::cout << "Mean of data1:\n" << mean1 << std::endl;
    // std::cout << "Variance of data1:\n" << var1 << std::endl;

    // std::cout << "Mean of data2:\n" << mean2 << std::endl;
    // std::cout << "Variance of data2:\n" << var2 << std::endl;

    // // 计算样本数
    // int64_t n1 = data1.size(0);
    // int64_t n2 = data2.size(0);
    // int64_t n_total = n1 + n2;

    // // 按照公式融合均值和方差
    // auto mean_total = (n1 * mean1 + n2 * mean2) / n_total;

    // auto var_total_internal = ((n1-0) * var1 + (n2-0) * var2) / (n_total-0);
    // auto var_total_mean_diff = ((double)n1 * (double)n2 / (((double)n_total-0) * ((double)n_total-0))) * ((mean1 - mean2).unsqueeze(-1).matmul((mean1 - mean2).unsqueeze(-2)));
    // std::cout << "(mean1 - mean2)" << mean1 - mean2 << std::endl; 
    // std::cout << "(mean1 - mean2).unsqueeze(-1): " << (mean1 - mean2).unsqueeze(-1) << std::endl;// {3}  -> {3,1}
    // std::cout << "(mean1 - mean2).unsqueeze(-2): " << (mean1 - mean2).unsqueeze(-2) << std::endl;// {3}  -> {1,3}
    // std::cout << "((mean1 - mean2).unsqueeze(-1).matmul((mean1 - mean2).unsqueeze(-2))): " << ((mean1 - mean2).unsqueeze(-1).matmul((mean1 - mean2).unsqueeze(-2))) << std::endl;// {3,3}
    // std::cout << "var_total_internal: \n" << var_total_internal << std::endl;// {3,3}
    // std::cout << "var_total_mean_diff: \n" << var_total_mean_diff.squeeze() << std::endl;// {3,3}

    // auto var_total = var_total_internal + var_total_mean_diff.squeeze();

    // // 打印融合后的均值和方差
    // std::cout << "Fused Mean:\n" << mean_total << std::endl;
    // std::cout << "Fused Variance:\n" << var_total << std::endl;

    // // 将两个数据连接后直接计算均值和方差
    // auto data_total = torch::cat({data1, data2}, /*dim=*/0);
    // std::cout << "data_total size:" << data_total.sizes() << std::endl;
    // std::cout << "data_total:\n" << data_total << std::endl;
    // auto mean_total_direct = data_total.mean(/*dim=*/0);
    // auto var_total_direct = torch::cov(data_total.transpose(0, 1), /*correction=*/0);

    // // 打印直接计算的均值和方差
    // std::cout << "Direct Mean:\n" << mean_total_direct << std::endl;
    // std::cout << "Direct Variance:\n" << var_total_direct << std::endl;

    // // 对比两种方法的精度
    // auto mean_diff = (mean_total - mean_total_direct).abs().max().item<float>();
    // auto var_diff = (var_total - var_total_direct).abs().max().item<float>();

    // std::cout << "Maximum difference in mean: " << mean_diff << std::endl;
    // std::cout << "Maximum difference in variance: " << var_diff << std::endl;





    // //PART:4 batch compute matrix's eigenvalue and eigenvector ////////////////////////////////////////////////////////////////////////////////////////
    // std::cout << "=====PART:4 batch compute matrix's eigenvalue and eigenvector" << std::endl;
    // // 创建一个10x10x3x3的随机张量
    // // torch::Tensor tensor_matrixs = torch::rand({10, 10, 3, 3});
    // // 创建一个固定的3x3矩阵
    // std::cout << torch::rand({3, 3}) << std::endl;// 0~1 range  

    // torch::Tensor fixed_matrix = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}});
    
    // /*
    // matrix: 1 2 3 
    //         4 5 6
    //         7 8 9
    // evals:  1.6117e+01 -1.1168e+00  2.9486e-07
    // evecs:  -0.2320 -0.7858  0.4082
    //         -0.5253 -0.0868 -0.8165
    //         -0.8187  0.6123  0.4082
    // */
    // // 在第0维和第1维分别插入大小为1的维度，使其形状变为1x1x3x3
    // torch::Tensor unsqueezed_matrix = fixed_matrix.unsqueeze(0).unsqueeze(0);

    // // 使用expand将1x1x3x3扩展为10x10x3x3
    // torch::Tensor tensor_matrixs = unsqueezed_matrix.expand({10, 10, 3, 3}) + torch::rand({10, 10, 3, 3}) * 0.0;
    // std::cout << "tensor size:" << tensor_matrixs.sizes() << std::endl;
    // std::cout << "tensor[5][5]: \n" << tensor_matrixs[5][5] << std::endl;

    // // 将10x10x3x3的张量转换为100x3x3的张量
    // torch::Tensor batch_tensor = tensor_matrixs.view({100, 3, 3});
    // std::cout << "batch_tensor size:" << batch_tensor.sizes() << std::endl;

    // // 批量特征分解
    // auto [eigenvalues, eigenvectors] = torch::linalg_eig(batch_tensor); //NOTE: 100x3x3 batch compute eigenvalue and eigenvector of matrix
    
    // // 提取特征值和特征向量
    // std::cout << "eigenvalues size:" << eigenvalues.sizes() << std::endl; //size: 100 3
    // // std::cout << "eigenvalues: \n" << eigenvalues << std::endl;
    // std::cout << "eigenvectors size:" << eigenvectors.sizes() << std::endl;//size: 100 3 3
    // // std::cout << "eigenvectors: \n" << eigenvectors << std::endl;
    // // std::cout << "eigenvectors: \n" << eigenvectors << std::endl;

    // auto real_eigenvalues = torch::real(eigenvalues);
    // // 找到每个矩阵的最小特征值的索引
    // torch::Tensor min_indices = real_eigenvalues.argmin(1);  // 100的张量，表示每个矩阵的最小特征值的索引

    // // 根据索引提取对应的特征向量
    // torch::Tensor min_eigenvectors = eigenvectors.index({torch::indexing::Slice(), min_indices});

    // auto evecs_final = eigenvectors.view({10, 10, 3, 3});
    // std::cout << "evecs_final size:" << evecs_final.sizes() << std::endl;
    // std::cout << "evecs_final[5][5]: \n" << evecs_final[5][5] << std::endl;
    // std::cout << "evecs_final[9][9]: \n" << evecs_final[9][9] << std::endl;
    // std::cout << "evecs_final[1][9]: \n" << evecs_final[1][9] << std::endl;

    return 0;
}