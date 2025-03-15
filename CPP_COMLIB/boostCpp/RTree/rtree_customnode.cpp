#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// 定义自定义结构体 Node
struct Node {
    bg::model::point<double, 2, bg::cs::cartesian> location;
    Eigen::Vector2d mean;
    Eigen::Matrix2d cov;
    bool isOccupy;
    std::string data;

    // 构造函数
    Node(double x, double y, Eigen::Vector2d meanIn, Eigen::Matrix2d covIn , bool occupy, std::string info)
        : location(x, y), mean(meanIn), cov(covIn), isOccupy(occupy), data(std::move(info)) {}
};

// 特化 indexable，用于从 Node 提取几何对象
namespace boost::geometry::index {
    template<>
    struct indexable<Node> {
        typedef bg::model::point<double, 2, bg::cs::cartesian> result_type;
        const result_type& operator()(const Node& n) const { return n.location; }
    };
} // namespace boost::geometry::index

// 特化 equal_to，用于比较 Node 是否相等
namespace boost::geometry::index {
    template<>
    struct equal_to<Node> {
        bool operator()(const Node& a, const Node& b) const {
            return bg::equals(a.location, b.location) && a.isOccupy == b.isOccupy && a.data == b.data;
        }
    };
} // namespace boost::geometry::index

int main() {
    // 创建 R-tree，存储类型为 Node
    bgi::rtree<Node, bgi::quadratic<16>> rtree;

    // 插入一些节点
    rtree.insert(Node(1.0, 1.0, Eigen::Vector2d::Ones(), Eigen::Matrix2d::Ones(), true, "Point A"));
    rtree.insert(Node(2.0, 2.0, Eigen::Vector2d::Ones()*2, Eigen::Matrix2d::Ones()*2, false, "Point B"));
    rtree.insert(Node(3.0, 3.0, Eigen::Vector2d::Ones()*3, Eigen::Matrix2d::Ones()*3, true, "Point C"));

    // 查询范围内的节点
    std::vector<Node> result;
    bg::model::box<bg::model::point<double, 2, bg::cs::cartesian>> query_box(
        bg::model::point<double, 2, bg::cs::cartesian>(0, 0),
        bg::model::point<double, 2, bg::cs::cartesian>(2, 2)
    );
    rtree.query(bgi::intersects(query_box), std::back_inserter(result));

    // 输出查询结果
    for (const auto& node : result) {
        std::cout << "Point: (" << node.location.get<0>() << ", " << node.location.get<1>() << "), " <<
                "Mean: (" << node.mean[0] << ", " << node.mean[1] << "), " <<
                "Cov: " << node.cov << ", "
                  << "Occupied: " << node.isOccupy << ", Data: " << node.data << std::endl;
    }

    return 0;
}