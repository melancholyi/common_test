#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <iostream>
#include <vector>

int main() {
    // Create a point cloud and add some random points
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.resize(1000);
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        cloud->points[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
        // cloud->points[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
        cloud->points[i].z = 512.0f; //NOTE: 2D search 
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;

    if (cloud->points.empty()) {
        std::cerr << "Error: Point cloud is empty!" << std::endl;
        return -1;
    }

    std::cout << "Original point cloud has " << cloud->points.size() << " points." << std::endl;

    // Create k-d tree and set input cloud
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    kdtree->setInputCloud(cloud);

    // Define query point and search radius
    pcl::PointXYZ searchPoint;
    searchPoint.x = 512.0f;
    searchPoint.y = 512.0f;
    searchPoint.z = 512.0f;
    float radius = 300.0f;

    std::cout << "Query point: (" << searchPoint.x << ", " << searchPoint.y << ", " << searchPoint.z << ")" << std::endl;

    // Store search results
    std::vector<int> pointIdxRadiusSearch;
    std::vector<float> pointRadiusSquaredDistance;

    if (kdtree->radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
        std::cout << "Found " << pointIdxRadiusSearch.size() << " points within radius " << radius << std::endl;
    } else {
        std::cout << "No points found within radius " << radius << std::endl;
        return 0;
    }

    // Visualize the original point cloud
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 255, 255, 255);
    viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original_cloud");

    // Highlight the query point
    pcl::PointCloud<pcl::PointXYZ>::Ptr queryPoint(new pcl::PointCloud<pcl::PointXYZ>);
    queryPoint->push_back(searchPoint);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> query_color(queryPoint, 255, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ>(queryPoint, query_color, "query_point");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "query_point");

    // Highlight the selected points within the radius
    pcl::PointCloud<pcl::PointXYZ>::Ptr selectedPoints(new pcl::PointCloud<pcl::PointXYZ>);
    for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
        selectedPoints->points.push_back((*cloud)[pointIdxRadiusSearch[i]]);
    }
    selectedPoints->width = selectedPoints->points.size();
    selectedPoints->height = 1;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> selected_color(selectedPoints, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ>(selectedPoints, selected_color, "selected_points");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "selected_points");

    // Draw a circle around the query point
    pcl::PointCloud<pcl::PointXYZ>::Ptr circlePoints(new pcl::PointCloud<pcl::PointXYZ>);
    for (float angle = 0; angle < 2 * M_PI; angle += M_PI / 180) {
        pcl::PointXYZ circlePoint;
        circlePoint.x = searchPoint.x + radius * cos(angle);
        circlePoint.y = searchPoint.y + radius * sin(angle);
        circlePoint.z = searchPoint.z;
        circlePoints->points.push_back(circlePoint);
    }
    circlePoints->width = circlePoints->points.size();
    circlePoints->height = 1;
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> circle_color(circlePoints, 0, 0, 255);
    viewer->addPointCloud<pcl::PointXYZ>(circlePoints, circle_color, "circle");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "circle");

    viewer->spinOnce(100000);

    // // Start the visualization loop
    // while (!viewer->wasStopped()) {
    //     viewer->spinOnce(100);
    // }

    return 0;
}