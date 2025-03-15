/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-01 11:27:27
 * @FilePath: /mesh_planner/test/compareCpp/multisetAndPQ/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <set>
#include <queue>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>
class MinHeap {
    private:
        std::vector<int> heap;
    
        // Function to heapify down the heap
        void heapifyDown(int index) {
            int smallest = index;
            int left = 2 * index + 1;
            int right = 2 * index + 2;
    
            if (left < heap.size() && heap[left] < heap[smallest]) {
                smallest = left;
            }
    
            if (right < heap.size() && heap[right] < heap[smallest]) {
                smallest = right;
            }
    
            if (smallest != index) {
                std::swap(heap[index], heap[smallest]);
                heapifyDown(smallest);
            }
        }
    
        // Function to heapify up the heap
        void heapifyUp(int index) {
            if (index == 0) return;
    
            int parent = (index - 1) / 2;
            if (heap[parent] > heap[index]) {
                std::swap(heap[parent], heap[index]);
                heapifyUp(parent);
            }
        }
    
    public:
        // Insert a new element into the heap
        void push(int value) {
            heap.push_back(value);
            heapifyUp(heap.size() - 1);
        }
    
        // Get the minimum element (root of the heap)
        int top() const {
            if (heap.empty()) {
                throw std::out_of_range("Heap is empty");
            }
            return heap[0];
        }
    
        // Remove the minimum element from the heap
        void pop() {
            if (heap.empty()) {
                throw std::out_of_range("Heap is empty");
            }
            heap[0] = heap.back();
            heap.pop_back();
            heapifyDown(0);
        }
    
        // Check if the heap is empty
        bool empty() const {
            return heap.empty();
        }
    
        // Print the heap
        void print() const {
            for (int value : heap) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
};

// test custom miniheap
void test_customMinHeap() {
    MinHeap heap;
    std::vector<int> data(10000);

    // 生成随机数据
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < 10000; ++i) {
        data[i] = rand() % 100000;
    }

    // 插入操作
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        heap.push(data[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> insert_time = end - start;
    std::cout << "customMinHeap insert time: " << insert_time.count() << " ms" << std::endl;

    // 访问最小元素并删除
    start = std::chrono::high_resolution_clock::now();
    while (!heap.empty()) {
        double temp = heap.top();
        heap.pop();
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> access_and_delete_time = end - start;
    std::cout << "\ncustomMinHeap access and delete time: " << access_and_delete_time.count() << " ms" << std::endl;
    std::cout << "sum time: " << insert_time.count() + access_and_delete_time.count() << " ms" << std::endl;
}



// 测试 std::multiset 的性能
void test_multiset() {
    std::multiset<int> ms;
    std::vector<int> data(10000);

    // 生成随机数据
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < 10000; ++i) {
        data[i] = rand() % 100000;
    }

    // 插入操作
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        ms.insert(data[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> insert_time = end - start;
    std::cout << "std::multiset insert time: " << insert_time.count() << " ms" << std::endl;

    // 访问最小元素并删除
    start = std::chrono::high_resolution_clock::now();
    while (!ms.empty()) {
        double min = *ms.begin();
        // std::cout << *ms.begin() << " ";
        ms.erase(ms.begin());
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> access_and_delete_time = end - start;
    std::cout << "\nstd::multiset access and delete time: " << access_and_delete_time.count() << " ms" << std::endl;
    std::cout << "sum time: " << insert_time.count() + access_and_delete_time.count() << " ms" << std::endl;
}

// 测试 std::priority_queue 的性能
void test_priority_queue() {
    std::priority_queue<int, std::vector<int>, std::greater<int>> pq;
    std::vector<int> data(10000);

    // 生成随机数据
    srand(static_cast<unsigned>(time(0)));
    for (int i = 0; i < 10000; ++i) {
        data[i] = rand() % 100000;
    }

    // 插入操作
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        pq.push(data[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> insert_time = end - start;
    std::cout << "std::priority_queue insert time: " << insert_time.count() << " ms" << std::endl;

    // 访问最小元素并删除
    start = std::chrono::high_resolution_clock::now();
    while (!pq.empty()) {
        double min = pq.top();
        // std::cout << pq.top() << " ";
        pq.pop();
    }
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> access_and_delete_time = end - start;
    std::cout << "\nstd::priority_queue access and delete time: " << access_and_delete_time.count() << " ms" << std::endl;
    std::cout << "sum time: " << insert_time.count() + access_and_delete_time.count() << " ms" << std::endl;   
}

int main() {
    std::cout << "==========Testing std::multiset:" << std::endl;
    test_multiset();

    std::cout << "\n==========Testing std::priority_queue:" << std::endl;
    test_priority_queue();

    std::cout << "\n==========Testing custom miniheap:" << std::endl;
    test_customMinHeap();

    return 0;
}