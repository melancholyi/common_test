/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-27 05:30:58
 * @FilePath: /mesh_planner/test/boostCpp/BTree/BTree.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <iostream>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

int main() {
    // Define the point type
    typedef bg::model::point<double, 2, bg::cs::cartesian> point_t;

    // Create an R-tree with a custom value type (e.g., pair of point and data)
    typedef std::pair<point_t, std::string> value_t;
    bgi::rtree<value_t, bgi::quadratic<16>> rtree;

    // Insert some points into the R-tree
    rtree.insert(std::make_pair(point_t(1, 1), "Point A"));
    rtree.insert(std::make_pair(point_t(2, 2), "Point B"));
    rtree.insert(std::make_pair(point_t(3, 3), "Point C"));

    // Query the R-tree for points within a range
    std::vector<value_t> result;
    point_t lower(0, 0), upper(2, 2);
    bg::model::box<point_t> query_box(lower, upper);
    rtree.query(bgi::intersects(query_box), std::back_inserter(result));

    // Output the results
    for (const auto& val : result) {
        std::cout << "Point: (" << val.first.get<0>() << ", " << val.first.get<1>() << ") - Data: " << val.second << std::endl;
    }

    return 0;
}