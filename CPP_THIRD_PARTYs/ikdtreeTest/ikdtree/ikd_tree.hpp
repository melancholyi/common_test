/*
Description: ikd-Tree: an incremental k-d tree for robotic applications 
Author: Yixi Cai
email: yixicai@connect.hku.hk
*/

/*
 * @Author: chasey melancholycy@gmail.com
 * @Date: 2025-03-01 11:13:21
 * @FilePath: /mesh_planner/test/customLibTest/ikdtree/ikd_tree/ikd_tree.hpp
 * @Description: This file modified by chasey, mainly change some interface(function and type) name
 * 
 * Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
 */

#pragma once
#include <stdio.h>
#include <queue>
#include <pthread.h>
#include <chrono>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#include <memory>
#include <pcl/point_types.h>

#define EPSS 1e-6
#define Minimal_Unbalanced_Tree_Size 10
#define Multi_Thread_Rebuild_Point_Num 1500
#define DOWNSAMPLE_SWITCH true
#define ForceRebuildPercentage 0.2
#define Q_LEN 1000000

using namespace std;

// struct ikdTree_PointType
// {
//     float x,y,z;
//     ikdTree_PointType (float px = 0.0f, float py = 0.0f, float pz = 0.0f){
//         x = px;
//         y = py;
//         z = pz;
//     }
// };

struct BoxPointType{
    float vertex_min[3];
    float vertex_max[3];
};

enum operation_set {ADD_POINT, DELETE_POINT, DELETE_BOX, ADD_BOX, DOWNSAMPLE_DELETE, PUSH_DOWN};

enum delete_point_storage_set {NOT_RECORD, DELETE_POINTS_REC, MULTI_THREAD_REC};

template <typename T>
class MANUAL_Q{
    private:
        int head = 0,tail = 0, counter = 0;
        T q[Q_LEN];
        bool is_empty;
    public:
        void pop();
        T front();
        T back();
        void clear();
        void push(T op);
        bool empty();
        int size();
};



template<typename PointType>
class IKDTree{
public:
    using PointVector = vector<PointType, Eigen::aligned_allocator<PointType>>;
    using Ptr = shared_ptr<IKDTree<PointType>>;
    struct IKDTreeNode{
        PointType point;
        uint8_t division_axis;  
        int TreeSize = 1;
        int invalid_point_num = 0;
        int down_del_num = 0;
        bool point_deleted = false;
        bool tree_deleted = false; 
        bool point_downsample_deleted = false;
        bool tree_downsample_deleted = false;
        bool need_push_down_to_left = false;
        bool need_push_down_to_right = false;
        bool working_flag = false;
        float radius_sq;
        pthread_mutex_t push_down_mutex_lock;
        float node_range_x[2], node_range_y[2], node_range_z[2];   
        IKDTreeNode *left_son_ptr = nullptr;
        IKDTreeNode *right_son_ptr = nullptr;
        IKDTreeNode *father_ptr = nullptr;
        // For paper data record
        float alpha_del;
        float alpha_bal;
    };

    struct Operation_Logger_Type{
        PointType point;
        BoxPointType boxpoint;
        bool tree_deleted, tree_downsample_deleted;
        operation_set op;
    };

    struct PointType_CMP{
        PointType point;
        float dist = 0.0;
        PointType_CMP (PointType p = PointType(), float d = INFINITY){
            this->point = p;
            this->dist = d;
        };
        bool operator < (const PointType_CMP &a)const{
            if (fabs(dist - a.dist) < 1e-10) return point.x < a.point.x;
            else return dist < a.dist;
        }    
    };

    class MANUAL_HEAP{
        public:
            MANUAL_HEAP(int max_capacity = 100){
                cap = max_capacity;
                heap = new PointType_CMP[max_capacity];
                heap_size = 0;
            }

            ~MANUAL_HEAP(){ delete[] heap;}

            void pop(){
                if (heap_size == 0) return;
                heap[0] = heap[heap_size-1];
                heap_size--;
                MoveDown(0);
                return;
            }

            PointType_CMP top(){ return heap[0];}

            void push(PointType_CMP point){
                if (heap_size >= cap) return;
                heap[heap_size] = point;
                FloatUp(heap_size);
                heap_size++;
                return;
            }

            int size(){ return heap_size;}

            void clear(){ heap_size = 0;}
        private:
            int heap_size = 0;
            int cap = 0;        
            PointType_CMP * heap;
            void MoveDown(int heap_index){
                int l = heap_index * 2 + 1;
                PointType_CMP tmp = heap[heap_index];
                while (l < heap_size){
                    if (l + 1 < heap_size && heap[l] < heap[l+1]) l++;
                    if (tmp < heap[l]){
                        heap[heap_index] = heap[l];
                        heap_index = l;
                        l = heap_index * 2 + 1;
                    } else break;
                }
                heap[heap_index] = tmp;
                return;
            }
            
            void FloatUp(int heap_index){
                int ancestor = (heap_index-1)/2;
                PointType_CMP tmp = heap[heap_index];
                while (heap_index > 0){
                    if (heap[ancestor] < tmp){
                        heap[heap_index] = heap[ancestor];
                        heap_index = ancestor;
                        ancestor = (heap_index-1)/2;
                    } else break;
                }
                heap[heap_index] = tmp;
                return;
            }

    };    

private:
    // Multi-thread Tree Rebuild
    bool termination_flag = false;
    bool rebuild_flag = false;
    pthread_t rebuild_thread;
    pthread_mutex_t termination_flag_mutex_lock, rebuild_ptr_mutex_lock, working_flag_mutex, search_flag_mutex;
    pthread_mutex_t rebuild_logger_mutex_lock, points_deleted_rebuild_mutex_lock;
    // queue<Operation_Logger_Type> Rebuild_Logger;
    MANUAL_Q<Operation_Logger_Type> Rebuild_Logger;    
    PointVector Rebuild_PCL_Storage;
    IKDTreeNode ** Rebuild_Ptr = nullptr;
    int search_mutex_counter = 0;
    static void * multi_thread_ptr(void *arg);
    void multi_thread_rebuild();
    void start_thread();
    void stop_thread();
    void run_operation(IKDTreeNode ** root, Operation_Logger_Type operation);
    // KD Tree Functions and augmented variables
    int Treesize_tmp = 0, Validnum_tmp = 0;
    float alpha_bal_tmp = 0.5, alpha_del_tmp = 0.0;
    float delete_criterion_param = 0.5f;
    float balance_criterion_param = 0.7f;
    float downsample_size = 0.2f;
    bool Delete_Storage_Disabled = false;
    IKDTreeNode * STATIC_ROOT_NODE = nullptr;
    PointVector Points_deleted;
    PointVector Downsample_Storage;
    PointVector Multithread_Points_deleted;
    void InitTreeNode(IKDTreeNode * root);
    void Test_Lock_States(IKDTreeNode *root);
    void BuildTree(IKDTreeNode ** root, int l, int r, PointVector & Storage);
    void Rebuild(IKDTreeNode ** root);
    int Delete_by_range(IKDTreeNode ** root, BoxPointType boxpoint, bool allow_rebuild, bool is_downsample);
    void Delete_by_point(IKDTreeNode ** root, PointType point, bool allow_rebuild);
    void Add_by_point(IKDTreeNode ** root, PointType point, bool allow_rebuild, int father_axis);
    void Add_by_range(IKDTreeNode ** root, BoxPointType boxpoint, bool allow_rebuild);
    void Search(IKDTreeNode * root, int k_nearest, PointType point, MANUAL_HEAP &q, double max_dist);//priority_queue<PointType_CMP>
    void Search_by_range(IKDTreeNode *root, BoxPointType boxpoint, PointVector &Storage);
    void Search_by_radius(IKDTreeNode *root, PointType point, float radius, PointVector &Storage);
    bool Criterion_Check(IKDTreeNode * root);
    void Push_Down(IKDTreeNode * root);
    void Update(IKDTreeNode * root); 
    void delete_tree_nodes(IKDTreeNode ** root);
    void downsample(IKDTreeNode ** root);
    bool same_point(PointType a, PointType b);
    float calc_dist(PointType a, PointType b);
    float calc_box_dist(IKDTreeNode * node, PointType point);    
    static bool point_cmp_x(PointType a, PointType b); 
    static bool point_cmp_y(PointType a, PointType b); 
    static bool point_cmp_z(PointType a, PointType b); 

public:
    //! constructor and destructor, and initialization
    IKDTree(float delete_param = 0.5, float balance_param = 0.6 , float box_length = 0.2);
    ~IKDTree();
    void setDeleteCriterionParam(float delete_param);
    void setBalanceCriterionParam(float balance_param);
    void setDownsampleParam(float box_length);
    void initIKDTree(float delete_param = 0.5, float balance_param = 0.7, float box_length = 0.2); 
    
    //! helper functions
    int size();
    int validnum();
    void rootAlpha(float &alpha_bal, float &alpha_del);
    
    //! build 
    void build(PointVector point_cloud);
    
    //! search
    void knnSearch(PointType point, int k_nearest, PointVector &Nearest_Points, vector<float> & Point_Distance, double max_dist = INFINITY);
    void boxSearch(const BoxPointType &Box_of_Point, PointVector &Storage);
    void radiusSearch(PointType point, const float radius, PointVector &Storage);
    
    //! insert
    int insertPoints(PointVector & PointToAdd, bool downsample_on);
    void insertBoxes(vector<BoxPointType> & BoxPoints);
    
    //! delete
    void deletePoints(PointVector & PointToDel);
    int deleteBoxes(vector<BoxPointType> & BoxPoints);

    void flatten(IKDTreeNode * root, PointVector &Storage, delete_point_storage_set storage_type);
    void acquireRemovedPoints(PointVector & removed_points);
    BoxPointType getRange();
    PointVector PCL_Storage;     
    IKDTreeNode * Root_Node = nullptr;
    int max_queue_size = 0;
};
