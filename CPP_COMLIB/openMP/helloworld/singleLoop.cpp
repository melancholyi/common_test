/*
 * @Author: chasey && melancholycy@gmail.com
 * @Date: 2025-04-12 02:53:03
 * @LastEditTime: 2025-04-15 13:41:30
 * @FilePath: /test/CPP_COMLIB/openMP/helloworld/singleLoop.cpp
 * @Description: 
 * @Reference: 
 * Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
 */
#include <iostream>
#include <unordered_map>
#include <string>
#include <omp.h>
#include <mutex>
#include <vector>

std::mutex my_map_mutex;

const int COUNT = 10;
const int NUM_THREADS = 3;

int main() {
    std::unordered_map<int, std::string> my_map;
    for(int i = 0; i < COUNT ; i ++){
        my_map[i] = "Thread " + std::to_string(i);
    }
    // #pragma omp parallel for collapse(2)
    omp_set_num_threads(NUM_THREADS); //NOTE: limit threads num
    #pragma omp parallel for //NOTE: for parallel   
    for (int i = 0; i < COUNT; i++) {
        std::lock_guard<std::mutex> lock(my_map_mutex);//NOTE: mutex lock
        auto it = my_map.find(i);
        if (it != my_map.end()) {
            
            std::cout << "Thread " << omp_get_thread_num() << " found: " << it->second << std::endl;
            my_map.erase(it);
        }
    }

    //////////////////////////// vec1 test //////////////////////////////////
    std::cout << "\n==== vec1 dynamic alloc memory with vec.push_back()====" << std::endl;
    std::vector<int> vec1;
    #pragma omp parallel for //NOTE: for parallel   
    for (int i = 0; i < COUNT; i++) {
        vec1.push_back(i);
    }

    // print vec1
    
    for(int i = 0; i < COUNT ; i ++){
        std::cout << "data: " << vec1[i] << std::endl;
    }

    ///////////////////////////////// vec2 test //////////////////////////////////
    std::cout << "\n==== vec2 prealloc memory with vec.push_back() ====" << std::endl;
    //! correct  
    std::vector<int> vec2;
    vec2.reserve(COUNT);
    //NOTE: prealloc memory 
    #pragma omp parallel for //NOTE: for parallel, dynamic alloc memory     
    for (int i = 0; i < COUNT; i++) {
        vec2.push_back(i);
    }

    // print vec2
    for(int i = 0; i < COUNT ; i ++){
        std::cout << "data: " << vec2[i] << std::endl;
    }

    ///////////////////////////////// vec3 test //////////////////////////////////
    std::cout << "\n==== vec3 prealloc memory with vec[i] = i ====" << std::endl;
    std::vector<float> vec3;
    vec3.resize(COUNT);
    //NOTE: prealloc memory 
    #pragma omp parallel for //NOTE: for parallel, dynamic alloc memory     
    for (int i = 0; i < COUNT; i++) {
        vec3[i] = i;
    }

    // print vec3
    std::cout << "vec3.size(): " << vec3.size() << std::endl;
    for(int i = 0; i < COUNT ; i ++){
        std::cout << "data: " << vec3[i] << std::endl;
    }


    return 0;

    return 0;
}



/*这样更倾向于将整个循环均匀分配给线程数， 下面这个案例 30 个循环 8 个线程， 所以大部分线程负责 4 个循环， 但是有的线程只负责 3 个循环，并且负责的循环是临近的
一般不要设置线程数太多，如果不设置|omp_set_num_threads(8)|。 omp会给每一个循环都分配一个线程， 这样会导致线程数过多， 造成线程切换的开销， 反而降低了性能
|omp_set_num_threads(8)|  |#pragma omp parallel|  |for + std::lock_guard<std::mutex> lock(my_map_mutex)|

Thread 0 found: Thread 0
Thread 0 found: Thread 1
Thread 0 found: Thread 2
Thread 0 found: Thread 3

Thread 3 found: Thread 12
Thread 3 found: Thread 13
Thread 3 found: Thread 14
Thread 3 found: Thread 15

Thread 6 found: Thread 24
Thread 6 found: Thread 25
Thread 6 found: Thread 26

Thread 2 found: Thread 8
Thread 2 found: Thread 9
Thread 2 found: Thread 10
Thread 2 found: Thread 11

Thread 1 found: Thread 4
Thread 1 found: Thread 5
Thread 1 found: Thread 6
Thread 1 found: Thread 7

Thread 4 found: Thread 16
Thread 4 found: Thread 17
Thread 4 found: Thread 18
Thread 4 found: Thread 19

Thread 5 found: Thread 20
Thread 5 found: Thread 21
Thread 5 found: Thread 22
Thread 5 found: Thread 23

Thread 7 found: Thread 27
Thread 7 found: Thread 28
Thread 7 found: Thread 29
*/