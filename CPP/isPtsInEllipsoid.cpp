/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-04 14:32:41
 * @FilePath: /mesh_planner/test/cpp/isPtsInEllipsoid.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <cmath>
#include <vector>

// 将角度转换为弧度
double deg2rad(double degrees) {
    return degrees * M_PI / 180.0;
}

double rad2deg(double radians) {
    return radians * 180.0 / M_PI;
}

// 判断点是否在椭圆内
bool isPtInEllipsoid(double pt_x, double pt_y, double center_x, double center_y, 
                     double aSquaInv, double bSquaInv, double theta) {
    // 将点的坐标转换为相对于椭圆中心的坐标
    double x_centered = pt_x - center_x;
    double y_centered = pt_y - center_y;

    // 将角度从度转换为弧度
    double theta_rad = deg2rad(theta);

    // 旋转坐标，将点旋转到椭圆的主轴方向
    double x_rotated = x_centered * cos(theta_rad) + y_centered * sin(theta_rad);
    double y_rotated = -x_centered * sin(theta_rad) + y_centered * cos(theta_rad);

    // 使用椭圆的标准方程判断点是否在椭圆内
    return (x_rotated * x_rotated * aSquaInv) + (y_rotated * y_rotated * bSquaInv) <= 1;
}

int main() {
    // 定义栅格参数
    double grid_min_x = 0.0, grid_min_y = 0.0;
    double grid_max_x = 2.0, grid_max_y = 2.0;
    double resolution = 0.25;

    // 定义椭圆参数
    double ellipse_center_x = 1.0, ellipse_center_y = 1.0;
    double semi_major_axis = 1.0;  // 半长轴
    double semi_minor_axis = 0.5;  // 半短轴
    double rotation_angle = 30.0;   // 旋转角度（单位：度）

    // 计算 a^2 和 b^2 的倒数
    double aSquaInv = 1.0 / (semi_major_axis * semi_major_axis);
    double bSquaInv = 1.0 / (semi_minor_axis * semi_minor_axis);

    // 创建栅格
    int num_x = static_cast<int>((grid_max_x - grid_min_x) / resolution);
    int num_y = static_cast<int>((grid_max_y - grid_min_y) / resolution);
    std::vector<std::string> strs; 
    for (int i = 0; i < num_y; ++i) {
        std::string line;
        for (int j = 0; j < num_x; ++j) {
            // 栅格的左下角坐标
            double grid_x = grid_min_x + j * resolution;
            double grid_y = grid_min_y + i * resolution;
            // 栅格中心点坐标
            double grid_center_x = grid_x + resolution / 2.0;
            double grid_center_y = grid_y + resolution / 2.0;

            // 判断栅格中心点是否在椭圆内
            if (isPtInEllipsoid(grid_center_x, grid_center_y, 
                                ellipse_center_x, ellipse_center_y, 
                                aSquaInv, bSquaInv, rotation_angle)) {
                line += "1 ";
            } else {
                line += "0 ";
            }
        }
        line += "\n";
        strs.push_back(line);
    }

    for(int i = strs.size() - 1; i > 0 ; i--){
        std::cout << strs[i] ;
    }

    return 0;
}