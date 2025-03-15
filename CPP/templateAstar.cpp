#include <iostream>
#include <vector>
#include <memory>
#include <queue>
#include <functional>
#include <Eigen/Dense>

// 基础的PathNode结构体
template <typename T>
struct PathNode {
    Eigen::VectorXd position;  // 位置信息
    T extraData;               // 用于扩展的额外数据
    PathNode* parent = nullptr;
    double gCost = 0.0;        // 到起点的实际代价
    double hCost = 0.0;        // 启发式代价

    // 用于比较的函数
    bool operator>(const PathNode& other) const {
        return (gCost + hCost) > (other.gCost + other.hCost);
    }
};


template <typename Node>
class AStar {
public:
    using NodeSPtr = std::shared_ptr<Node>;

    AStar() = default;
    virtual ~AStar() = default;

    void getNeighbors(const NodeSPtr& cur, std::vector<NodeSPtr>& exts){
        std::cout << "cur.position: " << cur->position << std::endl;
        std::cout << "cur.extraData: " << cur->extraData << std::endl;
        std::cout << "cur.gCost: " << cur->gCost << std::endl;
        std::cout << "cur.hCost: " << cur->hCost << std::endl;
    }
};


int main(){
    auto node_cur = std::make_shared<PathNode<Eigen::VectorXd>>();
    node_cur->position = Eigen::Vector3d(1,2,3);
    node_cur->extraData = Eigen::Vector3d(10,20,30);
    node_cur->gCost = 10;
    node_cur->hCost = 10;

    auto astar = AStar<PathNode<Eigen::VectorXd>>(); 
    std::vector<std::shared_ptr<PathNode<Eigen::VectorXd>>> exts;
    astar.getNeighbors(node_cur, exts);



    return 0;
}