/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-01 11:12:54
 * @FilePath: /mesh_planner/test/customLibTest/ikdtreeTest/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-01 11:12:54
 * @FilePath: /mesh_planner/test/customLibTest/ikdtreeTest/main.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <random>
#include <chrono>
#include <vector>
#include "pcl/point_types.h"
#include "pcl/point_cloud.h"
// #include <ikd_Tree.h>
#include "ikdtree/ikd_tree.hpp"  

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// 定义点类型
typedef bg::model::point<double, 2, bg::cs::cartesian> point_t;

#define POINT_NUM 100000
// KD_TREE<pcl::PointXYZ> ikdtree(0.5, 0.6, 0.2);
IKDTree<pcl::PointXYZ> ikdtree;

int main() {
    std::cout << "run here 1111 !!!" << std::endl;
    // 创建 RTree
    bgi::rtree<point_t, bgi::quadratic<16>> rtree;
    std::cout << "run here 1111 !!!" << std::endl;
    
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //NOTE: generate 100000 points 
    // 随机生成 100000 个点并插入 RTree
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1000.0, 1000.0);
    std::vector<point_t> points, rmvPoints;
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> cloud, rmvCloud;

    
    for (size_t i = 0; i < POINT_NUM; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        point_t p(x, y);
        points.push_back(p);
        rmvPoints.push_back(point_t(x + 2000, y + 2000));


        pcl::PointXYZ pt;
        pt.x = x;   
        pt.y = y;
        pt.z = 0.0;
        cloud.push_back(pt);

        pcl::PointXYZ pt_rmv;
        pt_rmv.x = x + 2000;
        pt_rmv.y = y + 2000;
        pt_rmv.z = 0.0;
        rmvCloud.push_back(pt_rmv);
    }

    std::cout <<  POINT_NUM << " points generated over." << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! NOTE: insert points into rtree and ikdtree
    //! insert points into rtree
    auto start1 = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < POINT_NUM; ++i) {
        rtree.insert(points[i]);
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> rtree_insert_time = end1 - start1;
    std::cout << "===== INSERT ===== RTree   insert time: " << rtree_insert_time.count() << " ms" << std::endl;

    //! insert points into ikdtree
    auto start2 = std::chrono::high_resolution_clock::now();
    ikdtree.build(cloud);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ikdtree_insert_time = end2 - start2;
    std::cout << "===== INSERT ===== IKDTree insert time: " << ikdtree_insert_time.count() << " ms" << std::endl;

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 执行半径搜索
    point_t query_point(0.0, 0.0); // 查询点
    double radius = 10.0;          // 查询半径

    std::vector<point_t> results;
    auto start4 = std::chrono::high_resolution_clock::now();
    rtree.query(bgi::satisfies([&query_point, radius](point_t const& p) {
        return bg::distance(query_point, p) < radius;
    }), std::back_inserter(results));
    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> rtree_radius_search_time = end4 - start4;
    std::cout << "===== SEARCH ===== RTree   radius search time: " << rtree_radius_search_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== RTree  radius search results: " << results.size() << " points found within radius " << radius << "." << std::endl;
    for(auto pt: results){
        std::cout << " " << bg::get<0>(pt) << " " << bg::get<1>(pt) << std::endl;
    }

    pcl::PointXYZ query_point2;
    query_point2.x = 0.0;
    query_point2.y = 0.0;
    query_point2.z = 0.0;
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> ikd_results;
    auto start3 = std::chrono::high_resolution_clock::now();
    ikdtree.radiusSearch(query_point2, radius, ikd_results);
    auto end3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ikdtree_radius_search_time = end3 - start3;
    std::cout << "===== SEARCH ===== IKDTree radius search time: " << ikdtree_radius_search_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== IKDTree radius search results: " << ikd_results.size() << " points found within radius " << radius << "." << std::endl;
    for(auto pt: ikd_results){
        std::cout << " " << pt.x << " " << pt.y << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //! remove points from rtree and ikdtree (un in it)
    std::cout << "\n===== REMOVE THE UNEXIST POINTS =====" << std::endl;
    auto start7 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POINT_NUM; ++i) {
        rtree.remove(rmvPoints[i]);
    }
    auto end7 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> rtree_remove_nopts_time = end7 - start7;
    std::cout << "===== REMOVE ===== RTree remove time: " << rtree_remove_nopts_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== RTree removed over, size: " << rtree.size() << std::endl;
    
    auto start8 = std::chrono::high_resolution_clock::now();
    ikdtree.deletePoints(rmvCloud);
    auto end8 = std::chrono::high_resolution_clock::now();
    auto ikdtree_remove_nopts_time = end8 - start8;
    std::cout << "===== REMOVE ===== IKDTree remove time: " << ikdtree_remove_nopts_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== IKDTree removed over, size: " << ikdtree.size() << std::endl;



    //////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "\n===== REMOVE THE EXIST POINTS =====" << std::endl;
    auto start5 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < POINT_NUM; ++i) {
        rtree.remove(points[i]);
    }
    auto end5 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> rtree_remove_time = end5 - start5;
    std::cout << "===== REMOVE ===== RTree remove time: " << rtree_remove_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== RTree removed over, size: " << rtree.size() << std::endl;

    auto strart6 = std::chrono::high_resolution_clock::now();
    ikdtree.deletePoints(cloud);
    auto end6 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ikdtree_remove_time = end6 - strart6;
    std::cout << "===== REMOVE ===== IKDTree remove time: " << ikdtree_remove_time.count() << " ms" << std::endl;
    std::cout << "===== PRINTF ===== IKDTree removed over, size: " << ikdtree.size() << std::endl;
    return 0;
}