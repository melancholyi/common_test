

#include <iostream>
#include <vector>
#include <queue>
#include <functional>
#include <cmath>
#include <unordered_map>
#include <memory>
#include <any>
#include <chrono>

// Eigen
#include <Eigen/Core>

#define PI_X_2 6.283185307179586
//! node type
enum class eType {
    OPEN_E = 0, CLOSE_E, UNEXPAND_E,
};

//! path node
template<int STATE_DIM>  
struct PathNode{
    using IndexType = Eigen::Matrix<int, STATE_DIM, 1>;
    using StateType = Eigen::Matrix<double, STATE_DIM, 1>;
    using SPtr = std::shared_ptr<PathNode>;
    static constexpr int DIM = STATE_DIM;

    StateType state;
    IndexType index;
    eType type;
    double f, g, h;
    SPtr parent;

    PathNode(){
        reset();    
    }

    void reset(){
        state.setZero();
        index.setZero();
        type = eType::UNEXPAND_E;
        f = g = h = 0.0;
        parent = nullptr;
    }

    void state2index(){
        const double EPS_INV = 1e3;
        index = state.unaryExpr([EPS_INV](double x) {
            return static_cast<int>(std::floor(x * EPS_INV));
        });
    }
};

//! NodeHashTable 
template <typename NodePtrType, int DIM>
class NodeHashTable 
{
    private:
        using IdxType = Eigen::Matrix<int, DIM, 1>;
        template <typename T>
        struct matrix_hash : std::unary_function<T, size_t> {
            std::size_t operator()(T const& matrix) const {
                size_t seed = 0;
                for (long int i = 0; i < matrix.size(); ++i) {
                    auto elem = *(matrix.data() + i);
                    seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                return seed;
            }
        };

        std::unordered_map<IdxType, NodePtrType, matrix_hash<IdxType>> data_;

    public:
        NodeHashTable() {}
        ~NodeHashTable() {}

        void insert(const IdxType& idx, const NodePtrType& node){
            data_.insert(std::make_pair(idx, node));
        }

        NodePtrType find(const IdxType& idx){
            auto iter = data_.find(idx);
            return iter == data_.end() ? NULL : iter->second;
        }

        void clear() { data_.clear(); }
};

template<typename NodeType>
class AStarSearch {
    public:
        using NodeSPtrType = std::shared_ptr<NodeType>;
    private:
        struct CompareNode {
            bool operator()(const NodeSPtrType& a, const NodeSPtrType& b) const {
                return a->f > b->f;  
            }
        };
        using PriorQueueType = std::priority_queue<NodeSPtrType, std::vector<NodeSPtrType>, CompareNode>;
        using NodeHashTableType = NodeHashTable<NodeSPtrType, NodeType::DIM>;

    public:
        //PART: construction  
        AStarSearch(const bool& isUsePtr = true, const size_t& preAllocSize = 1000): isUsePtr_(isUsePtr), preAllocSize_(preAllocSize), nodeNum_(0){
            for (size_t i = 0; i < preAllocSize_; i++){  
                nodeSPtrPool_.push_back(std::make_shared<NodeType>());
            }
            earlyExitFun_ = [](const NodeSPtrType& curNode, const NodeSPtrType& goalNode){
                if((curNode->state - goalNode->state).norm() < 0.1){
                    return true;
                }else{
                    return false;
                }
            };
            visExtNodesFun_ = [](const std::vector<NodeSPtrType>& extFreeNodes){
                return;
            };
        }

        //PART: std::function interface functions  
        std::function<void(const NodeSPtrType&, std::vector<NodeSPtrType>&)> getExtendNodesFun_;
        std::function<double(const NodeSPtrType&, const NodeSPtrType&)> getExtendCostFun_;
        std::function<double(const NodeSPtrType&, const NodeSPtrType&)> getHeuristicCostFun_;
        std::function<bool(const NodeSPtrType&)> isOccupyFun_;
        std::function<void(const std::vector<NodeSPtrType>&)> visExtNodesFun_; 
        std::function<bool(const NodeSPtrType&, const NodeSPtrType&)> earlyExitFun_;

        inline NodeSPtrType getNodeSPtrFromPool(){
            if(nodeNum_ == preAllocSize_){
                for (size_t i = 0; i < preAllocSize_; i++){  
                    nodeSPtrPool_.push_back(std::make_shared<NodeType>());
                }
                preAllocSize_ *= 2;
            }
            return nodeSPtrPool_[nodeNum_++];
        }
        
        //PART: main functions   
        bool plan(const typename NodeType::StateType& start, const typename NodeType::StateType& goal, std::vector<typename NodeType::StateType>& path){
            //! reset  
            PriorQueueType empty;
            openlist_.swap(empty);
            openset_.clear();
            for(size_t i = 0; i < nodeNum_; i++){
                nodeSPtrPool_[i]->reset();
            }
            nodeNum_ = 0;

            //! start node & goal node    
            NodeSPtrType start_node = getNodeSPtrFromPool();
            start_node->state = start;
            start_node->state2index();
            start_node->type = eType::OPEN_E;

            NodeSPtrType goal_node = getNodeSPtrFromPool();
            goal_node->state = goal;
            goal_node->state2index();

            //! check /*start*/ and goal is all free
            if(isUsePtr_){
                if(/*isOccupyFun_(start) ||*/ isOccupyFun_(goal_node)){
                    std::cout << "goal isn't free!!!" << std::endl;
                    return false;
                }
            }else{
                if(/*isOccupy(start) ||*/ isOccupy(goal_node)){
                    return false;
                }
            }

            //! begin plan  
            openlist_.push(start_node);
            openset_.insert(start_node->index, start_node);

            //! while(!openlist_.empty())
            while (!openlist_.empty()){
                //! get the mini-cost node  
                NodeSPtrType cur_node = openlist_.top();
                openlist_.pop();
                cur_node->type = eType::CLOSE_E;

                //! early exit  
                bool is_exit = (isUsePtr_ ? earlyExitFun_(cur_node, goal_node) : earlyExit(cur_node, goal_node));
                if(is_exit){
                    retrievePath(cur_node, path);
                    std::cout << "AT_GOAL" << std::endl;
                    return true;
                }

                //! generate ext_states and ext_nodes    
                // std::vector<typename NodeType::StateType> ext_states;
                std::vector<NodeSPtrType> ext_nodes, ext_visnodes;
                if(isUsePtr_){
                    getExtendNodesFun_(cur_node, ext_nodes);
                }else{
                    getExtendNodes(cur_node, ext_nodes);
                }

                //! foreach ext_nodes  
                for(auto ext_node : ext_nodes){
                    bool is_occ = isUsePtr_ ? isOccupyFun_(ext_node) : isOccupy(ext_node);
                    if(is_occ){
                        continue;
                    }

                    ext_visnodes.push_back(ext_node);
                    ext_node->state2index();
                    auto ext_ptr = openset_.find(ext_node->index);
                    if(ext_ptr != nullptr && ext_ptr->type == eType::CLOSE_E){// find it 
                        continue;
                    }else{
                        auto ext_g = cur_node->g + (isUsePtr_ ? getExtendCostFun_(cur_node, ext_node) : getExtendCost(cur_node, ext_node));
                        auto ext_h = isUsePtr_ ? getHeuristicCostFun_(cur_node, goal_node) : getHeuristicCost(cur_node, goal_node);
                        auto ext_f = ext_g + ext_h;

                        //UNEXPAND_E type
                        if(ext_ptr == nullptr){
                            ext_ptr = getNodeSPtrFromPool();
                            ext_ptr->state = ext_node->state;
                            ext_ptr->state2index();
                            ext_ptr->type = eType::OPEN_E;
                            ext_ptr->g = ext_g;
                            ext_ptr->h = ext_h;
                            ext_ptr->f = ext_f;
                            ext_ptr->parent = cur_node;

                            openlist_.push(ext_ptr);
                            openset_.insert(ext_ptr->index, ext_ptr);

                        }else if(ext_ptr != nullptr && ext_ptr->type == eType::OPEN_E){
                            if(ext_g < ext_ptr->g){
                                ext_ptr->state = ext_node->state;
                                ext_ptr->state2index();
                                ext_ptr->g = ext_g;
                                ext_ptr->h = ext_h;
                                ext_ptr->f = ext_f;
                                ext_ptr->parent = cur_node;
                            }
                        }else{
                            //pass
                        }
                    }
                }
                visExtNodesFun_(ext_visnodes);
            }

            return true;
        }

        void printTest() {
            std::cout << "NodeType::DIM: " << NodeType::DIM << std::endl;

            //========== Test openlist
            NodeSPtrType node1 = std::make_shared<NodeType>(); node1->f = 10;
            NodeSPtrType node2 = std::make_shared<NodeType>(); node2->f = 40;
            NodeSPtrType node3 = std::make_shared<NodeType>(); node3->f = 50;
            NodeSPtrType node4 = std::make_shared<NodeType>(); node4->f = 20;
            NodeSPtrType node5 = std::make_shared<NodeType>(); node5->f = 30;

            openlist_.push(node1);
            openlist_.push(node2);
            openlist_.push(node3);
            openlist_.push(node4);
            openlist_.push(node5);

            while (!openlist_.empty()) {
                std::cout << "top()->f: " << openlist_.top()->f << std::endl;
                openlist_.pop();
            }

            //========== 
            node1->state = Eigen::Vector3d(1.1234, 1.1234, 1.1234); node1->state2index();
            node2->state = Eigen::Vector3d(2.1234, 2.1234, 2.1234); node2->state2index();
            node3->state = Eigen::Vector3d(3.1234, 3.1234, 3.1234); node3->state2index();
            node4->state = Eigen::Vector3d(4.1234, 4.1234, 4.1234); node4->state2index();
            node5->state = Eigen::Vector3d(5.1234, 5.1234, 5.1234); node5->state2index();

            openset_.insert(node1->index, node1);
            openset_.insert(node2->index, node2);
            openset_.insert(node3->index, node3);
            openset_.insert(node4->index, node4);
            openset_.insert(node5->index, node5);

            auto temptr1 = openset_.find(node1->index);
            auto temptr2 = openset_.find(node2->index);
            auto temptr3 = openset_.find(node3->index);
            auto temptr4 = openset_.find(node4->index);
            auto temptr5 = openset_.find(node5->index);
            std::cout << temptr1->f << std::endl;
            std::cout << temptr2->f << std::endl;
            std::cout << temptr3->f << std::endl;
            std::cout << temptr4->f << std::endl;
            std::cout << temptr5->f << std::endl;

            std::cout << temptr1->state.size() << std::endl;
            std::cout << temptr1->index.size() << std::endl;
        }


    private://membership functions
        //PART: virtual functions  
        virtual void getExtendNodes(const NodeSPtrType& curNode, std::vector<NodeSPtrType>& extStates){}
        virtual double getExtendCost(const NodeSPtrType& curNode, const NodeSPtrType& extNode){return 0.0;}
        virtual double getHeuristicCost(const NodeSPtrType& extNode, const NodeSPtrType& goalNode){return 0.0;}
        virtual bool isOccupy(const NodeSPtrType& curNode){return false;}
        virtual bool earlyExit(const NodeSPtrType& curNode, const NodeSPtrType& goalNode){
            bool flag = (curNode->state - goalNode->state).norm() < 0.1;
            return flag;
        }


        //PART: help functions  
        void state2index(const Eigen::VectorXd& state, Eigen::VectorXi& index){
            const double EPS_INV = 1e3;
            index = (state.array() * EPS_INV).floor().cast<int>();
        }
        inline void retrievePath(const NodeSPtrType& endNode, std::vector<typename NodeType::StateType>& path){
            NodeSPtrType cur_node = endNode;
            path.push_back(cur_node->state);
            while (cur_node->parent != NULL)
            {
                cur_node = cur_node->parent;
                path.push_back(cur_node->state);
            }
            std::reverse(path.begin(), path.end());
            return;
        }


    private://membership variables  
        bool isUsePtr_;
        size_t preAllocSize_;
        size_t nodeNum_;
        std::vector<NodeSPtrType> nodeSPtrPool_;
        PriorQueueType openlist_;
        NodeHashTableType openset_;
    
    public:
        using SPtr = std::shared_ptr<AStarSearch>;
};

void testVectorOfObjects(int numNodes) {
    std::vector<PathNode<3>> nodes;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numNodes; ++i) {
        PathNode<3> node;
        node.f = i;
        nodes.push_back(node); // 将对象复制到 vector 中
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time to push back " << numNodes << "   objects directly: " << duration.count()*1000 << " ms\n";
}

void testVectorOfSharedPtrs(int numNodes) {
    std::vector<std::shared_ptr<PathNode<3>>> nodes;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numNodes; ++i) {
        auto node = std::make_shared<PathNode<3>>(); // 在堆上创建对象
        node->f = i;
        nodes.push_back(node); // 将智能指针存储到 vector 中
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time to push back " << numNodes << " shared_ptr objects: " << duration.count()*1000 << " ms\n";
}


int main() {
    auto node = PathNode<3>();
    node.state = Eigen::Vector3d(1.1234, 23.3245345, 123.12309);
    node.state2index();
    std::cout << "Node dimension: " << node.DIM << std::endl;
    std::cout << "Node dimension: " << node.index.transpose() << std::endl;


    AStarSearch<PathNode<3>> astar;
    astar.printTest();
    auto state = node.state;
    PathNode<3>::StateType state_test;
    state_test(0) = 1;
    auto state_copy = state_test;



    ////////////////////////////
    //compare time at stack and heap
    constexpr int numNodes = 10000;

    std::cout << "Testing vector of objects...\n";
    testVectorOfObjects(numNodes);

    std::cout << "Testing vector of shared_ptr objects...\n";
    testVectorOfSharedPtrs(numNodes);

    ////////////////////////////////
    using GridAstarType = AStarSearch<PathNode<2>>;
    GridAstarType astar_grid;
    astar_grid.getExtendNodesFun_ = [](const GridAstarType::NodeSPtrType& curNode, std::vector<GridAstarType::NodeSPtrType>& extNodes){
            //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(1,0,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(-1,0,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(0,-1,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(0, 1,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(-1,-1,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(-1,1,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(1,-1,0)));
    //   extSE2s.push_back((utils::SE2Type)(cur.array() + Eigen::Array3d(1, 1,0)));

        GridAstarType::NodeSPtrType temp1 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp2 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp3 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp4 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp5 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp6 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp7 = std::make_shared<PathNode<2>>();
        GridAstarType::NodeSPtrType temp8 = std::make_shared<PathNode<2>>();
        
        temp1->state = curNode->state.array() + Eigen::Array2d(-1, 0);
        temp2->state = curNode->state.array() + Eigen::Array2d( 1, 0);
        temp3->state = curNode->state.array() + Eigen::Array2d( 0,-1);
        temp4->state = curNode->state.array() + Eigen::Array2d( 0, 1);
        temp5->state = curNode->state.array() + Eigen::Array2d(-1, -1);
        temp6->state = curNode->state.array() + Eigen::Array2d( 1, 1);
        temp7->state = curNode->state.array() + Eigen::Array2d( 1,-1);
        temp8->state = curNode->state.array() + Eigen::Array2d(-1, 1);
        extNodes.push_back(temp1);
        extNodes.push_back(temp2);
        extNodes.push_back(temp3);
        extNodes.push_back(temp4);
        extNodes.push_back(temp5);
        extNodes.push_back(temp6);
        extNodes.push_back(temp7);
        extNodes.push_back(temp8);
    };

    astar_grid.getExtendCostFun_ = [](const GridAstarType::NodeSPtrType& curNode, const GridAstarType::NodeSPtrType& extNode){
        return (curNode->state - extNode->state).norm();
    };

    astar_grid.getHeuristicCostFun_ = [](const GridAstarType::NodeSPtrType& curNode, const GridAstarType::NodeSPtrType& goalNode){
        return (curNode->state - goalNode->state).norm();
    };

    astar_grid.isOccupyFun_ = [](const GridAstarType::NodeSPtrType& curNode){
        return false;
    };

    astar_grid.visExtNodesFun_ = [](const std::vector<GridAstarType::NodeSPtrType>& extNode){

    };

    std::vector<Eigen::Vector2d> path;
    astar_grid.plan(Eigen::Vector2d(0,0), Eigen::Vector2d(10,10), path);
    std::cout << "Path points:\n";
    for (const auto& point : path) {
        std::cout << "Point: (" << point.x() << ", " << point.y() << ")\n";
    }





    // std::function<double(const NodeSPtrType&, const NodeSPtrType&)> getExtendCostFun_;
    // std::function<double(const NodeSPtrType&, const NodeSPtrType&)> getHeuristicCostFun_;
    // std::function<bool(const NodeSPtrType&)> isOccupyFun_;

    return 0;
}