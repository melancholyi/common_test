/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2024-12-19 12:51:25
 * @FilePath: /mesh_planner/test/cpp/unorderedmap.cpp
 * @Description: 
 * 
 * Copyright (c) 2024 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <memory>
#include <unordered_map>
#include <list>
#include <Eigen/Dense>
#include <set>

struct Voxel {
    float x, y, z;
};

struct SpaceHashFunctor {
    inline size_t operator()(const Eigen::Array3i& key) const {
        return size_t(((key[0]) * static_cast<size_t>(73856093)) ^
                      ((key[1]) * static_cast<size_t>(471943)) ^
                      ((key[2]) * static_cast<size_t>(83492791))) %
                  static_cast<size_t>(1000000000);
    }
};

struct EqualWithFunctor {
    inline bool operator()(const Eigen::Array3i& a, const Eigen::Array3i& b) const {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }
};

using SpaceHashMap = std::unordered_map<
    Eigen::Array3i,
    typename std::list<std::pair<Eigen::Array3i, std::unique_ptr<Voxel>>>::iterator,
    SpaceHashFunctor,
    EqualWithFunctor>;

void printVoxel(const Voxel& voxel) {
    std::cout << "Voxel at (" << voxel.x << ", " << voxel.y << ", " << voxel.z << ")\n";
}




struct GridHashFunctor {
    inline size_t operator()(const std::pair<int,int>& xy) const {
        return size_t( ((xy.first) * long(73856093)) 
                    ^((xy.second) * long(83492791))) 
                    % size_t(1000000000);
    }
// inline size_t operator()(const int& x, const int& y) const {
//   return size_t( (x * long(73856093)) 
//                 ^(y * long(83492791))) 
//                 % size_t(1000000000);
// } 
// inline size_t operator()(const Eigen::Vector2i& xy) const {
//   return size_t( (xy(0) * long(73856093)) 
//                 ^(xy(1) * long(83492791))) 
//                 % size_t(1000000000);
// } 
};
struct GridEqualFunctor {  
    bool operator()(const std::pair<int, int> &lhs, const std::pair<int, int> &rhs) const {
        return lhs.first == rhs.first && lhs.second == rhs.second;
    }
};
struct Gaussian3D{
    Eigen::Vector3d mean;
    Eigen::Matrix3d cov;
};


// Define the custom comparator
struct CompareByMeanZ {
    bool operator()(const Gaussian3D& a, const Gaussian3D& b) const {
        return a.mean(2) < b.mean(2);
    }
};



/**
 * @description: 
 * @attendion: 
 * @return {*}
 */
int main() {
    SpaceHashMap spaceHashMap;

    // Create a voxel to insert
    Voxel voxel = {1.0f, 2.0f, 3.0f};
    Eigen::Array3i key(0, 0, 0); // The key for the voxel

    // Search for the voxel in the map
    auto it = spaceHashMap.find(key);
    if (it == spaceHashMap.end()) {
        // If not found, insert the voxel into the map
        std::list<std::pair<Eigen::Array3i, std::unique_ptr<Voxel>>> list;
        list.emplace_back(key, std::make_unique<Voxel>(voxel));
        auto listIt = list.begin();
        spaceHashMap[key] = listIt;
        std::cout << "Inserted new voxel at key (" << key[0] << ", " << key[1] << ", " << key[2] << ")\n";
    } else {
        // If found, print the voxel
        auto temp = it->second;
        auto& voxelPair = it->second->second;
        printVoxel(*voxelPair);
        std::cout << "Found voxel at key (" << key[0] << ", " << key[1] << ", " << key[2] << ")\n";
    }

    // Now let's search for another voxel with a different key
    Eigen::Array3i anotherKey(1, 1, 1);
    it = spaceHashMap.find(anotherKey);
    if (it == spaceHashMap.end()) {
        // If not found, insert a new voxel into the map
        Voxel anotherVoxel = {4.0f, 5.0f, 6.0f};
        std::list<std::pair<Eigen::Array3i, std::unique_ptr<Voxel>>> list;
        list.emplace_back(anotherKey, std::make_unique<Voxel>(anotherVoxel));
        auto listIt = list.begin();
        spaceHashMap[anotherKey] = listIt;
        std::cout << "Inserted new voxel at key (" << anotherKey[0] << ", " << anotherKey[1] << ", " << anotherKey[2] << ")\n";
    } else {
        // If found, print the voxel
        auto& voxelPair = it->second->second;
        printVoxel(*voxelPair);
        std::cout << "Found voxel at key (" << anotherKey[0] << ", " << anotherKey[1] << ", " << anotherKey[2] << ")\n";
    }

    std::list<double> double_list;
    // double_list..em

    ///////////////////////////////////////////////////////////////////////////
    std::unordered_map<std::pair<int, int>, Gaussian3D, GridHashFunctor, GridEqualFunctor> elevMap;  
    // Example data
    Eigen::Vector3d mean1(1.0, 2.0, 3.0);
    Eigen::Matrix3d cov1(3, 3);
    cov1 << 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;

    Eigen::Vector3d mean2(4.0, 5.0, 6.0);
    Eigen::Matrix3d cov2(3, 3);
    cov2 << 2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 2.0;
    
    // Insert data into the map
    elevMap[{0, 0}] = {mean1, cov1};
    elevMap[{1, 1}] = {mean2, cov2};

    // Retrieve and print data
    for (const auto& entry : elevMap) {
        std::cout << "Grid: (" << entry.first.first << ", " << entry.first.second << ")\n";
        std::cout << "Mean:\n" << entry.second.mean << "\n";
        std::cout << "Covariance:\n" << entry.second.cov << "\n\n";
    }

    ////////////////////////////////////////////////////////////////////
    // Create the set with the custom comparator
    std::set<Gaussian3D, CompareByMeanZ> dataset;

    // Example data
    Gaussian3D g1 = {Eigen::Vector3d(1.0, 2.0, 3.0), Eigen::Matrix3d::Identity()};
    Gaussian3D g2 = {Eigen::Vector3d(4.0, 5.0, 6.0), Eigen::Matrix3d::Identity()};
    Gaussian3D g3 = {Eigen::Vector3d(7.0, 8.0, 9.0), Eigen::Matrix3d::Identity()};

    // Insert data into the set
    

    dataset.insert(g3);
    dataset.insert(g2);
    dataset.insert(g1);

    // Print the sorted data
    for (const auto& data : dataset) {
        std::cout << "Mean: " << data.mean.transpose() << "\n";
        std::cout << "Covariance:\n" << data.cov << "\n\n";
    }
    

    return 0;
}