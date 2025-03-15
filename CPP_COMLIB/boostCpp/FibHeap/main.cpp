/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-25 10:39:37
 * @FilePath: /mesh_planner/test/boostCpp/FibHeap/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <boost/heap/fibonacci_heap.hpp>
#include <iostream>

// 前向声明 Node 类
class Node;

// 定义比较函数
struct CompareNode {
    bool operator()(Node* lhs, Node* rhs) const;
};

// 定义 Fibonacci 堆类型
typedef boost::heap::fibonacci_heap<Node*, boost::heap::compare<CompareNode>> FibonacciHeap;
typedef FibonacciHeap::handle_type HeapHandle;

// 定义 Node 类
class Node {
public:
    int fScore; // 用于比较的分数
    HeapHandle handle; // 用于存储堆中的句柄

    Node(int score) : fScore(score) {}

    // 获取 fScore 的值
    int getFScore() const { return fScore; }
};

// 实现比较函数
bool CompareNode::operator()(Node* lhs, Node* rhs) const {
    return lhs->fScore > rhs->fScore;
}

int main() {
    // 创建 Fibonacci 堆
    FibonacciHeap heap;

    // 创建一些 Node 对象
    Node node1(10);
    Node node2(5);
    Node node3(15);

    // 将 Node 对象插入堆中
    node1.handle = heap.push(&node1);
    Node* topNode1 = heap.top();
    std::cout << "Top node fScore: " << topNode1->fScore << std::endl;
    
    node2.handle = heap.push(&node2);
    Node* topNode2 = heap.top();
    std::cout << "Top node fScore: " << topNode2->fScore << std::endl;

    node3.handle = heap.push(&node3);
    Node* topNode3 = heap.top();
    std::cout << "Top node fScore: " << topNode3->fScore << std::endl;

    // 提取堆顶元素
    Node* topNode4 = heap.top();
    std::cout << "Top node fScore: " << topNode4->fScore << std::endl;

    // 弹出堆顶元素
    heap.pop();

    // 更新堆中元素的值
    node1.fScore = 3;
    heap.update(node1.handle);

    // 再次提取堆顶元素
    topNode4 = heap.top();
    std::cout << "New top node fScore: " << topNode4->fScore << std::endl;

    return 0;
}