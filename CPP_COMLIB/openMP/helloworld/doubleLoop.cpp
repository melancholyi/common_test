#include <iostream>
#include <vector>
#include <omp.h>
#include <mutex>



std::mutex my_map_mutex;

// 并行处理函数
void process_map_parallel(std::vector<bool>& map, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            // #pragma omp critical ////NOTE: 会花费大量时间
            // {
                map[index] = true; // 确保每个位置都被赋值
            // }

            // 打印线程信息
            // std::lock_guard<std::mutex> lock(my_map_mutex);//NOTE: 会花费大量时间
            // std::cout << "T:" << omp_get_thread_num() << " I:(" << row << ", " << col << ")" << std::endl;
        }
    }
}

// 串行处理函数
void process_map_serial(std::vector<bool>& map, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            map[index] = true; // 确保每个位置都被赋值
        }
    }
}

int main() {
    const int rows = 10;
    const int cols = 10;
    const int NUM_THREADS = 8; // 设置线程数量

    // 并行版本
    std::vector<bool> map_parallel(rows * cols, false); // 一维 vector 模拟二维地图
    map_parallel.reserve(rows * cols); // 提前分配足够的内存

    double start_time_parallel = omp_get_wtime(); // 记录开始时间
    omp_set_num_threads(NUM_THREADS);
    process_map_parallel(map_parallel, rows, cols); // 调用并行处理函数
    double end_time_parallel = omp_get_wtime(); // 记录结束时间
    double elapsed_time_parallel = end_time_parallel - start_time_parallel; // 计算执行时间

    // 验证地图（并行版本）
    int false_count_par = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            if (!map_parallel[index]) {
                false_count_par ++;
                // std::cerr << "Error (Parallel): Position (" << row << ", " << col << ") should be true" << std::endl;
            }

            // std::cout << (map_parallel[index] ? "X " : ". ");
        }
        // std::cout << std::endl;
    }
    

    // 串行版本
    std::vector<bool> map_serial(rows * cols, false); // 一维 vector 模拟二维地图
    map_serial.reserve(rows * cols); // 提前分配足够的内存

    double start_time_serial = omp_get_wtime(); // 记录开始时间
    process_map_serial(map_serial, rows, cols); // 调用串行处理函数
    double end_time_serial = omp_get_wtime(); // 记录结束时间
    double elapsed_time_serial = end_time_serial - start_time_serial; // 计算执行时间

    // 验证地图（串行版本）
    int false_count_ser = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            if (!map_serial[index]) {
                false_count_ser ++;
                std::cerr << "Error (Serial): Position (" << row << ", " << col << ") should be true" << std::endl;
            }
            
        }
    }


    // 打印时间对比
    std::cout << "Parallel time: " << elapsed_time_parallel << " seconds" << std::endl;
    std::cout << "Serial time:   " << elapsed_time_serial << " seconds" << std::endl;
    std::cout << "false_count_par: " << false_count_par << std::endl;
    std::cout << "false_count_ser: " << false_count_ser << std::endl;

    return 0;
}





// #include <iostream>
// #include <vector>
// #include <omp.h>
// #include <chrono>

// int main() {
//     const int NUM_THREADS = 8;
//     omp_set_num_threads(NUM_THREADS);
//     const int rows = 1000;  // 行数
//     const int cols = 1000;  // 列数
//     std::vector<bool> map(rows * cols, false);  // 初始化为false

//     // 串行版本
//     {
//         auto start = std::chrono::high_resolution_clock::now();
//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < cols; ++j) {
//                 int index = i * cols + j;
//                 map[index] = true;
//             }
//         }
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed = end - start;
//         std::cout << "  Serial time: " << elapsed.count() << " seconds" << std::endl;
//     }
//     // 验证结果
//     bool all_true = true;
//     for (bool val : map) {
//         if (!val) {
//             all_true = false;
//             break;
//         }
//     }
//     std::cout << "Serial All elements set to true: " << (all_true ? "Yes" : "No") << std::endl;

//     // 重置地图
//     std::fill(map.begin(), map.end(), false);

//     // 并行版本
//     {
//         auto start = std::chrono::high_resolution_clock::now();
//         #pragma omp parallel for collapse(2)
//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < cols; ++j) {
//                 int index = i * cols + j;
//                 map[index] = true;
//             }
//         }
//         auto end = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double> elapsed = end - start;
//         std::cout << "Parallel time: " << elapsed.count() << " seconds" << std::endl;
//     }

//     // 验证结果
//     bool all_true2 = true;
//     for (bool val : map) {
//         if (!val) {
//             all_true2 = false;
//             break;
//         }
//     }
//     std::cout << "Parallel All elements set to true: " << (all_true2 ? "Yes" : "No") << std::endl;

//     return 0;
// }