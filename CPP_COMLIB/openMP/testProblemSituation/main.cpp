#include <omp.h>
#include <iostream>
#include <vector>
#include <chrono>

void compareParallelAndSerialTwoLoop() {
    const int N = 10000;  // 增大矩阵大小以观察性能差异
    std::vector<int> matrix(N * N, 0);

    // 并行计算
    auto start1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = i + j;
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() 
              << " ms" << std::endl;

    // 串行计算
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i * N + j] = i + j;
        }
    }
    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Serial time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() 
              << " ms" << std::endl;

    // 输出矩阵（可选，对于大矩阵不建议输出）
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << matrix[i * N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
}

class compRuntime{
    private:
    std::chrono::_V2::system_clock::time_point start_, end_;


    public:
    compRuntime(){
        start_ = std::chrono::high_resolution_clock::now();
    }
    double getRuntimeMs(){
        end_ = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }
};


void parallelProblem(){
    const int N = 10000;
    uint64_t sum_parallel = 0, sum_serial = 0;

    // 并行化外层循环
    auto start1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for collapse(2) reduction(+:sum_parallel)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum_parallel += i + j;  // 多个线程同时修改sum，导致数据竞争
        }
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() 
              << " ms" << std::endl;

    auto start2 = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            sum_serial += i + j;  // 多个线程同时修改sum，导致数据竞争
        }
    }

    auto end2 = std::chrono::high_resolution_clock::now();
    std::cout << "Serial time: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() 
              << " ms" << std::endl;


    std::cout << "parallel computation sometimes will has problem!!!" << std::endl; 
    std::cout << "sum_parallel = " << sum_parallel  << std::endl;
    std::cout << "sum_serial   = " << sum_serial << std::endl;
}

int main() {
    compareParallelAndSerialTwoLoop();
    parallelProblem();
    return 0;
}