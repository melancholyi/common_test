/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-12 05:40:27
 * @LastEditTime: 2025-04-12 06:05:38
 * @FilePath: /test/CPP_COMLIB/openMP/helloworld/doubleLoopOther.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <vector>
#include <omp.h>
#include <mutex>

std::mutex my_map_mutex;

// 并行处理函数
void process_map_parallel(std::vector<int>& map, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    // #pragma omp parallel for
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            map[index] = 1; // 确保每个位置都被赋值

        }
    }
}

// 串行处理函数
void process_map_serial(std::vector<int>& map, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            map[index] = 1; // 确保每个位置都被赋值
        }
    }
}

int main() {
    const int rows = 4;
    const int cols = 4;
    const int NUM_THREADS = 8; // 设置线程数量

    // 并行版本
    std::vector<int> map_parallel(rows * cols, 0); // 一维 vector 模拟二维地图
    map_parallel.reserve(rows * cols); // 提前分配足够的内存

    double start_time_parallel = omp_get_wtime(); // 记录开始时间
    omp_set_num_threads(NUM_THREADS);
    process_map_parallel(map_parallel, rows, cols); // 调用并行处理函数
    double end_time_parallel = omp_get_wtime(); // 记录结束时间
    double elapsed_time_parallel = end_time_parallel - start_time_parallel; // 计算执行时间

    // 验证地图（并行版本）
    int zero_count_par = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            if (!map_parallel[index]) {
                zero_count_par ++;
                // std::cerr << "Error (Parallel): Position (" << row << ", " << col << ") should be 1" << std::endl;
            }

            // std::cout << (map_parallel[index] ? "X " : ". ");
        }
        // std::cout << std::endl;
    }
    

    // 串行版本
    std::vector<int> map_serial(rows * cols, 0); // 一维 vector 模拟二维地图
    map_serial.reserve(rows * cols); // 提前分配足够的内存

    double start_time_serial = omp_get_wtime(); // 记录开始时间
    process_map_serial(map_serial, rows, cols); // 调用串行处理函数
    double end_time_serial = omp_get_wtime(); // 记录结束时间
    double elapsed_time_serial = end_time_serial - start_time_serial; // 计算执行时间

    // 验证地图（串行版本）
    int zero_count_ser = 0;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            int index = row * cols + col;
            if (!map_serial[index]) {
                zero_count_ser ++;
                std::cerr << "Error (Serial): Position (" << row << ", " << col << ") should be 1" << std::endl;
            }
            
        }
    }


    // 打印时间对比
    std::cout << "Parallel time: " << elapsed_time_parallel << " seconds" << std::endl;
    std::cout << "Serial time:   " << elapsed_time_serial << " seconds" << std::endl;
    std::cout << "zero_count_par: " << zero_count_par << std::endl;
    std::cout << "zero_count_ser: " << zero_count_ser << std::endl;

    return 0;
}
