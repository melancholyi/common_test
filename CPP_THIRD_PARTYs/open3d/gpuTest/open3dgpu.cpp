/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-02-16 06:23:30
 * @FilePath: /test/CPP_THIRD_PARTYs/open3d/gpuTest/open3dgpu.cpp
 * @Description: 
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */
#include <iostream>
#include <open3d/Open3D.h>
class compRuntime{
    private:
    std::chrono::_V2::system_clock::time_point start_, end_;


    public:
    compRuntime(){
        start_ = std::chrono::high_resolution_clock::now();
    }
    double getRuntimeMs(){
        end_ = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
    }
};
int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Generate a simple synthetic point cloud
    std::vector<Eigen::Vector3d> points;
    for (double x = -1.0; x <= 10.0; x += 0.01) {
        for (double y = -1.0; y <= 10.0; y += 0.01) {
            double z = std::sin(x * x + y * y);
            points.emplace_back(x, y, z);
        }
    }

    // Create a legacy point cloud from the generated points
    auto pcd = std::make_shared<open3d::geometry::PointCloud>();
    pcd->points_ = points;

    // Define the GPU device
    open3d::core::Device device(open3d::core::Device::DeviceType::CUDA, 0);

    // Move the point cloud to the GPU
    open3d::t::geometry::PointCloud t_pcd = open3d::t::geometry::PointCloud::FromLegacy(*pcd);
    t_pcd = t_pcd.To(device);

    // Downsample the point cloud using voxel grid on the GPU
    float voxel_size = 0.1;
    auto downsampled = t_pcd.VoxelDownSample(voxel_size);
    open3d::t::geometry::TriangleMesh mesh = t_pcd.ComputeConvexHull();

    // Move the downsampled point cloud back to the CPU
    auto downsampled_legacy = downsampled.ToLegacy();

    // Wrap the downsampled point cloud in a shared pointer
    auto downsampled_legacy_ptr = std::make_shared<open3d::geometry::PointCloud>(downsampled_legacy);

    // Convert the Tensor-based mesh to legacy mesh for visualization
    open3d::geometry::TriangleMesh legacy_mesh = mesh.ToLegacy();
    auto mesh_ptr = std::make_shared<open3d::geometry::TriangleMesh>(legacy_mesh);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Total runtime: " << elapsed.count() << " ms\n";

    // // 可视化转换后的网格
    // open3d::visualization::DrawGeometries({mesh_ptr}, "Legacy TriangleMesh");

    // // Visualize the downsampled point cloud
    // open3d::visualization::DrawGeometries({downsampled_legacy_ptr}, "Downsampled Synthetic Point Cloud");

    return 0;
}


