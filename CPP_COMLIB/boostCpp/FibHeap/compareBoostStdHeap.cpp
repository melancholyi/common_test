/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-27 01:27:22
 * @FilePath: /mesh_planner/test/boostCpp/FibHeap/compareBoostStdHeap.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <boost/heap/fibonacci_heap.hpp>
#include <queue>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <set> // Include for std::multiset

using namespace std;
using namespace boost::heap;

// Function to generate random data
void generateRandomData(vector<int>& data, int size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(1, size); // Random numbers between 1 and 10000

    data.reserve(size);
    for (int i = 0; i < size; ++i) {
        data.push_back(dis(gen));
    }
}

// Test Boost's Fibonacci Heap
void testBoostFibHeap(const vector<int>& data) {
    using FibHeap = fibonacci_heap<int>;
    FibHeap heap;

    auto start1 = chrono::high_resolution_clock::now();
    for (int num : data) {
        heap.push(num);
    }
    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout << "===== PUSH ===== Boost Fibonacci Heap time: " << duration1.count() << " ms" << endl;

    auto start2 = chrono::high_resolution_clock::now();
    while (!heap.empty()) {
        heap.pop();
    }
    auto end2 = chrono::high_resolution_clock::now();

    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2);
    cout << "===== POP  ===== Boost Fibonacci Heap time: " << duration2.count() << " ms" << endl;
}

// Test standard priority_queue
void testStdPriorityQueue(const vector<int>& data) {
    priority_queue<int> pq;

    auto start1 = chrono::high_resolution_clock::now();
    for (int num : data) {
        pq.push(num);
    }

    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout << "===== PUSH ===== Standard priority_queue time: " << duration1.count() << " ms" << endl;

    auto start2 = chrono::high_resolution_clock::now();
    while (!pq.empty()) {
        pq.pop();
    }

    auto end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2);
    cout << "===== POP  ===== Standard priority_queue time: " << duration2.count() << " ms" << endl;

}


// Test std::multiset
void testStdMultiset(const vector<int>& data) {
    multiset<int> ms;

    auto start1 = chrono::high_resolution_clock::now();
    for (int num : data) {
        ms.insert(num);
    }
    auto end1 = chrono::high_resolution_clock::now();
    auto duration1 = chrono::duration_cast<chrono::milliseconds>(end1 - start1);
    cout << "===== INSERT ===== std::multiset time: " << duration1.count() << " ms" << endl;

    auto start2 = chrono::high_resolution_clock::now();
    while (!ms.empty()) {
        ms.erase(ms.begin());
    }
    auto end2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end2 - start2);
    cout << "===== ERASE ===== std::multiset time: " << duration2.count() << " ms" << endl;
}

int main() {
    const int dataSize = 1000000;
    vector<int> data;

    // Generate random data
    generateRandomData(data, dataSize);

    // Test Boost Fibonacci Heap
    cout << "Testing Boost Fibonacci Heap..." << endl;
    testBoostFibHeap(data);

    // Test standard priority_queue
    cout << "Testing Standard priority_queue..." << endl;
    testStdPriorityQueue(data);

    // Test std::multiset
    cout << "Testing std::multiset..." << endl;
    testStdMultiset(data);

    return 0;
}