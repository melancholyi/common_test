/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-18 03:57:21
 * @FilePath: /mesh_planner/test/cpp/unordered_set.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <unordered_set>
#include <vector>
#include <functional> // for std::hash

// 定义Point结构体
struct Point {
    int x;
    int y;

    // 构造函数
    Point(int x, int y) : x(x), y(y) {}

    // 重载等号运算符，用于比较两个点是否相等
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// 自定义哈希函数
namespace std {
    template <>
    struct hash<Point> {
        std::size_t operator()(const Point& p) const {
            // 使用std::hash对x和y分别哈希，然后组合成一个哈希值
            return std::hash<int>()(p.x) ^ (std::hash<int>()(p.y) << 1);
        }
    };
}

int main() {
    // 示例点集合
    std::vector<Point> points = {{1, 2}, {2, 3}, {1, 2}, {4, 5}, {2, 3}};

    // 使用unordered_set去重
    std::unordered_set<Point> uniquePoints(points.begin(), points.end());

    // 输出去重后的点
    std::cout << "Unique points:" << std::endl;
    for (const auto& point : uniquePoints) {
        std::cout << "(" << point.x << ", " << point.y << ")" << std::endl;
    }

    return 0;
}