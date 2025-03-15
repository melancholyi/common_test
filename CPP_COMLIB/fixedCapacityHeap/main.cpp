/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-01 11:49:18
 * @FilePath: /mesh_planner/test/customLibTest/fixedCapacityHeap/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <set>
#include <queue>
#include <chrono>

template <typename T, typename Compare = std::less<T>>
class FixedCapacityHeap {
public:
    FixedCapacityHeap(size_t capacity) : capacity_(capacity) {}

    void push(const T& value) {
        if (ms_.size() < capacity_) {
            ms_.insert(value);
        } else {
            auto it = ms_.end();
            it--;
            if (*it > value) {
                ms_.erase(it);
                ms_.insert(value);
            }
        }
    }

    bool empty() {
        return ms_.empty();
    }

    void pop() {
        if (!ms_.empty()) {
            ms_.erase(ms_.begin());
        }
    }

    T top() const {
        return *ms_.begin();
    }

    size_t size() const {
        return ms_.size();
    }

private:
    std::multiset<T, Compare> ms_;
    size_t capacity_;
};

int main() {
    FixedCapacityHeap<int> heap(3); // 固定容量为 3
    heap.push(10);
    heap.push(5);
    heap.push(3);
    heap.push(8);
    heap.push(1);
    heap.push(12);

    std::cout << "Heap size: " << heap.size() << std::endl;
    while (!heap.empty()) {
        std::cout << heap.top() << " ";
        heap.pop();
    }
    std::cout << std::endl;

    return 0;
}


// int main() {
//     const size_t capacity = 1000;
//     const size_t num_data = 2000;

//     FixedCapacityHeap<int> heap(capacity); // 固定容量为 1000
//     std::vector<int> data(num_data);

//     // 生成随机数据
//     srand(static_cast<unsigned>(time(0)));
//     for (size_t i = 0; i < num_data; ++i) {
//         data[i] = rand() % 100000;
//     }

//     // 插入操作
//     auto start = std::chrono::high_resolution_clock::now();
//     for (size_t i = 0; i < num_data; ++i) {
//         heap.push(data[i]);
//     }
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> insert_time = end - start;
//     std::cout << "Insert 2000 elements time: " << insert_time.count() << " ms" << std::endl;

//     // 访问和删除最小元素
//     start = std::chrono::high_resolution_clock::now();
//     while (!heap.empty()) {
//         auto temp = heap.top();
//         // std::cout << heap.top() << " ";
//         heap.pop();
//     }
//     end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double, std::milli> access_and_delete_time = end - start;
//     std::cout << "\nAccess and delete all elements time: " << access_and_delete_time.count() << " ms" << std::endl;

//     return 0;
// }