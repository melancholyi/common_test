/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-06 01:54:42
 * @FilePath: /utils_ws/test/autodiffTest/src/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <autodiff/forward/dual.hpp>
#include <chrono> // 用于时间测量

using namespace autodiff;
using namespace std::chrono;

dual f(dual x) {
    return 1.0 + x + x * x + 1.0 / x + log(x) + exp(x) + sin(x)*cos(x) + sqrt(x);
}
double df(double x){
    return 1 + 2*x -1/(x*x) + 1/x + exp(x) + cos(x)*cos(x) - sin(x)*sin(x) + 0.5 / sqrt(x);
}

int main() {
    double xdouble = 1.0;
    autodiff::dual x = 1.0;
    dual u = f(x);

    auto start_t1 = high_resolution_clock::now();
    double dudx = 0;
    for(int i = 0 ;i < 100000;i++){
        dudx = derivative(f, wrt(x), at(x));
    }
    
    auto stop_t1 = high_resolution_clock::now();
    auto duration_t1 = duration_cast<microseconds>(stop_t1 - start_t1);



    auto start_t2 = high_resolution_clock::now();
    double dfx = 0;
    for(int i = 0 ;i < 100000;i++){
        dfx = df(xdouble);   
    } 
    auto stop_t2 = high_resolution_clock::now();
    auto duration_t2 = duration_cast<microseconds>(stop_t2 - start_t2);

    // 输出运行时间
    std::cout << "autodiff runtime: " << duration_t1.count() << " us" << std::endl;
    std::cout << "handdiff runtime: " << duration_t2.count() << " us" << std::endl;

    std::cout << "u = " << u << std::endl;
    std::cout << "du/dx = " << dudx << std::endl;
    std::cout << "dfx = " << dfx << std::endl;

    /*  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
    
    autodiff runtime: 4697 us
    handdiff runtime: 0 us
    u = 8.17293
    du/dx = 5.80213
    dfx = 5.80213
    */
}