#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <chrono>
#include <pcl/console/time.h>
#include <pcl/filters/passthrough.h>
#include <filesystem>

int main(int argc, char** argv)
{


    // 获取输入文件路径和目录
    std::string input_pcd_path = "../data/ruins_pcd_all/map.pcd";
    std::filesystem::path path(input_pcd_path);
    std::string input_dir = path.parent_path().string();

    // 声明点云数据类型
    typedef pcl::PointXYZ PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    // 创建点云对象
    PointCloud::Ptr cloud(new PointCloud);
    PointCloud::Ptr cloud_filtered(new PointCloud);
    PointCloud::Ptr cloud_downsampled(new PointCloud);

    // 读取PCD文件
    if (pcl::io::loadPCDFile<PointT>(input_pcd_path, *cloud) == -1)
    {
        PCL_ERROR("无法读取文件 %s\n", input_pcd_path.c_str());
        return (-1);
    }
    std::cout << "已载入 " << cloud->width * cloud->height << " 个点" << std::endl;

    // 使用PassThrough过滤器剔除z坐标小于1.5的点
    pcl::PassThrough<PointT> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.81, 1000000.0);
    pass.filter(*cloud_filtered);

    std::cout << "剔除后剩下 " << cloud_filtered->width * cloud_filtered->height << " 个点" << std::endl;

    // 保存剔除后的点云到map_remove.pcd
    std::string remove_pcd_path = input_dir + "/map_remove.pcd";
    if (pcl::io::savePCDFileASCII(remove_pcd_path, *cloud_filtered) == -1)
    {
        PCL_ERROR("无法保存剔除后的点云到 %s\n", remove_pcd_path.c_str());
        return (-1);
    }

    // 体素栅格过滤器，用于降采样
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(cloud_filtered);
    sor.setLeafSize(0.1f, 0.1f, 0.1f);
    sor.filter(*cloud_downsampled);

    std::cout << "降采样后剩下 " << cloud_downsampled->width * cloud_downsampled->height << " 个点" << std::endl;

    // 保存降采样后的点云到map_dwz.pcd
    std::string dwz_pcd_path = input_dir + "/map_dwz.pcd";
    if (pcl::io::savePCDFileASCII(dwz_pcd_path, *cloud_downsampled) == -1)
    {
        PCL_ERROR("无法保存降采样后的点云到 %s\n", dwz_pcd_path.c_str());
        return (-1);
    }

    // 可视化（可选，注释掉这些代码可以不显示可视化窗口）
    
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("点云可视化"));
    viewer->addPointCloud(cloud_downsampled, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->initCameraParameters();

    viewer->spinOnce(5000);

    // while (!viewer->wasStopped())
    // {
    //     viewer->spinOnce(100);
    //     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // }
    

    std::cout << "处理完成，文件已保存到：" << input_dir << std::endl;
    return 0;
}
