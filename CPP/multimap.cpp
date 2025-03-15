/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2024-12-20 11:22:37
 * @FilePath: /mesh_planner/test/multimap.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <map>
#include <set>
    
#include <unordered_map>
#include <set>


using XY = std::pair<int, int>;
using ZSet = std::set<int>;

// Define a custom hash function for std::pair<int, int>
struct XYHash {
    std::size_t operator()(const XY& xy) const {
        return std::hash<int>()(xy.first) ^ std::hash<int>()(xy.second);
    }
};

// Define a custom equality function for std::pair<int, int>
struct XYEqual {
    bool operator()(const XY& lhs, const XY& rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};

std::unordered_map<XY, ZSet, XYHash, XYEqual> container;

void insertPoint(int x, int y, int z) {
    // Use emplace to avoid unnecessary set copying
    container[XY{x, y}].insert(z);
}

std::set<int> getZValues(int x, int y) {
    auto it = container.find(XY{x, y});
    if (it != container.end()) {
        return it->second;
    } else {
        // Return an empty set if the key is not found
        return ZSet();
    }
}

int main() {
    // 插入一些点
    insertPoint(1, 2, 3);
    insertPoint(1, 2, 5);
    insertPoint(1, 2, 1);
    insertPoint(1, 2, 3);
    insertPoint(1, 2, 5);
    insertPoint(1, 2, 1);

    insertPoint(3, 4, 6);
    insertPoint(3, 4, 8);

    // 获取x=1, y=2的所有z值
    auto z_values = getZValues(1, 2);
    for (int z : z_values) {
        std::cout << z << " ";
    }
    std::cout << std::endl;

    return 0;
}