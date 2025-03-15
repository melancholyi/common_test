#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <iostream>
#include <random>
#include <vector>
#include <thread>  // 包含 std::this_thread
#include <chrono>  // 包含 std::chrono

// 随机生成轨迹点
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomPath(int num_points, float range) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr path(new pcl::PointCloud<pcl::PointXYZ>);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-range, range);

    for (int i = 0; i < num_points; ++i) {
        pcl::PointXYZ point;
        point.x = distribution(generator);
        point.y = distribution(generator);
        point.z = distribution(generator);
        path->points.push_back(point);
    }
    return path;
}

// 在轨迹周围随机生成点云
pcl::PointCloud<pcl::PointXYZ>::Ptr generateRandomCloudAroundPath(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& path, 
    int num_points_per_path_point, 
    float radius) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-radius, radius);

    for (const auto& path_point : path->points) {
        for (int i = 0; i < num_points_per_path_point; ++i) {
            pcl::PointXYZ point;
            point.x = path_point.x + distribution(generator);
            point.y = path_point.y + distribution(generator);
            point.z = path_point.z + distribution(generator);
            cloud->points.push_back(point);
        }
    }
    return cloud;
}

// 提取轨迹周围的边界点云
void extractBoundaryPoints(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr& path,
                           pcl::PointCloud<pcl::PointXYZ>& boundary_points,
                           float radius) {
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(path);

    for (const auto& point : cloud->points) {
        std::vector<int> nearest_indices(1);
        std::vector<float> nearest_squared_distances(1);
        if (kdtree.nearestKSearch(point, 1, nearest_indices, nearest_squared_distances) > 0) {
            if (sqrt(nearest_squared_distances[0]) <= radius) {
                boundary_points.points.push_back(point);
            }
        }
    }
}

int main() {
    // 生成随机轨迹
    pcl::PointCloud<pcl::PointXYZ>::Ptr path = generateRandomPath(10, 1.0); // 生成10个轨迹点，范围在[-1, 1]

    // 在轨迹周围生成点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generateRandomCloudAroundPath(path, 100, 0.1); // 每个轨迹点周围生成100个点，半径为0.1

    // 提取轨迹周围的边界点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr boundary_points(new pcl::PointCloud<pcl::PointXYZ>);
    extractBoundaryPoints(cloud, path, *boundary_points, 0.1); // 提取距离轨迹小于0.1的点

    // 可视化
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // 显示轨迹点云（红色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> path_color_handler(path, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(path, path_color_handler, "path");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "path");

    // 显示生成的点云（蓝色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, cloud_color_handler, "cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cloud");

    // 显示提取的边界点云（绿色）
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> boundary_color_handler(boundary_points, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(boundary_points, boundary_color_handler, "boundary_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "boundary_points");

    // 启动可视化窗口
    viewer->initCameraParameters();
    viewer->spinOnce(1000000);
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }

    return 0;
}