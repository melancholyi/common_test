//! cpp
#include <iostream>
#include <random>
#include <chrono>
#include <vector>

//! pcl
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
#include "pcl/kdtree/kdtree_flann.h"

//! ikdtree
#include "ikdtree/ikd_tree.hpp"  


#define POINT_NUM 100000
IKDTree<pcl::PointXYZ> ikdtree1;
IKDTree<pcl::PointXYZ> ikdtree2;




int main() {
    //! random generate datas  
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> points;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < POINT_NUM; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        pcl::PointXYZ pt;
        pt.x = x;   
        pt.y = y;
        pt.z = 0.0;
        points.push_back(pt);
        cloud->points.push_back(pt);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    //================================BUILD V.S.=========================================
    //! KDTree build
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    auto start1 = std::chrono::high_resolution_clock::now();
    kdtree.setInputCloud(cloud);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kdtree_time = end1 - start1;
    std::cout << "=====  KDTREE BUILD ===== kdtree build time: " << kdtree_time.count() << " ms" << std::endl;

    //! IKDTree build
    auto start2 = std::chrono::high_resolution_clock::now();
    ikdtree1.build(points);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ikdtree_insert_time = end2 - start2;
    std::cout << "===== IKDTREE BUILD ===== IKDTree build time: " << ikdtree_insert_time.count() << " ms" << std::endl;

    //================================INCREAMENT INSERT V.S.=========================================
    const int LOOP_COUNT = 10;
    for(int i = 0 ; i < LOOP_COUNT; i++){
        

    }



}



