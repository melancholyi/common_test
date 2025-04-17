#include <iostream>
#include <vector>
#include <algorithm>
#include <utility> // 用于 std::pair
#include <omp.h>   // OpenMP 头文件
#include <chrono>  // 用于计时

// 网格参数
double grid_min_x = -1.0;
double grid_max_x = 1.0;
double grid_min_y = -1.0;
double grid_max_y = 1.0;
double grid_resolution = 0.1; // 网格分辨率（单位：米）

// 自定义网格类
class Grid {
private:
    std::vector<char> data_; // 使用 char 替代 bool
    int rows, cols;         // 网格的行数和列数

public:
    // 构造函数
    Grid(int r, int c) : rows(r), cols(c), data_(r * c, 0) {}

    // 拷贝构造函数
    Grid(const Grid& other) : rows(other.rows), cols(other.cols), data_(other.data_) {}

    // 拷贝赋值运算符
    Grid& operator=(const Grid& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data_ = other.data_;
        }
        return *this;
    }

    // 移动构造函数
    Grid(Grid&& other) noexcept : rows(other.rows), cols(other.cols), data_(std::move(other.data_)) {
        other.rows = 0;
        other.cols = 0;
    }

    // 移动赋值运算符
    Grid& operator=(Grid&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data_ = std::move(other.data_);
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    // 访问网格元素
    char& operator()(int row, int col) {
        return data_[row * cols + col];
    }

    const char& operator()(int row, int col) const {
        return data_[row * cols + col];
    }

    // 获取行数和列数
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // // 调整网格大小
    // void resize(int new_rows, int new_cols) {
    //     if (new_rows == rows && new_cols == cols) {
    //         return;
    //     }

    //     //拷贝旧数据到暂时的map然后再std::move到成员变量data_中
    //     std::vector<char> new_data(new_rows * new_cols, 0);
    //     #pragma omp parallel for collapse(2)
    //     for (int row = 0; row < std::min(rows, new_rows); ++row) {
    //         for (int col = 0; col < std::min(cols, new_cols); ++col) {
    //             new_data[row * new_cols + col] = data_[row * cols + col];
    //         }
    //     }
    //     data_ = std::move(new_data);
        
    //     rows = new_rows;
    //     cols = new_cols;
    // }

    void resize(int new_rows, int new_cols) {
        if (new_rows == rows && new_cols == cols) {
            return;
        }
    
        std::vector<char> new_data(new_rows * new_cols, 0);
        const int min_rows = std::min(rows, new_rows);
        const int min_cols = std::min(cols, new_cols);
        for (int row = 0; row < min_rows; ++row) {
            const char* src = &data_[row * cols];          // 旧数据的行起始位置
            char* dest = &new_data[row * new_cols];        // 新数据的行起始位置
            std::copy(src, src + min_cols, dest);           // 复制min_cols个元素
            // 或使用memcpy: 
            // std::memcpy(dest, src, min_cols * sizeof(char));
        }
        data_ = std::move(new_data);
        rows = new_rows;
        cols = new_cols;
    }

    // 更新点云范围
    void updatePointCloudRange(const std::vector<std::pair<double, double>>& points) {
        if (points.empty()) return;

        double current_min_x = points[0].first;
        double current_max_x = points[0].first;
        double current_min_y = points[0].second;
        double current_max_y = points[0].second;

        //#pragma omp parallel for collapse(2)
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            double x = points[i].first;
            double y = points[i].second;
            current_min_x = std::min(current_min_x, x);
            current_max_x = std::max(current_max_x, x);
            current_min_y = std::min(current_min_y, y);
            current_max_y = std::max(current_max_y, y);
        }

        grid_min_x = std::min(grid_min_x, current_min_x);
        grid_max_x = std::max(grid_max_x, current_max_x);
        grid_min_y = std::min(grid_min_y, current_min_y);
        grid_max_y = std::max(grid_max_y, current_max_y);
    }

    // 设置输入点云数据
    void setInput(const std::vector<std::pair<double, double>>& points) {
        //计算新的网格大小和范围然后计算新的行列数
        updatePointCloudRange(points);//update grid min and max range
        int new_rows = static_cast<int>((grid_max_x - grid_min_x) / grid_resolution) + 1;
        int new_cols = static_cast<int>((grid_max_y - grid_min_y) / grid_resolution) + 1;

        //设置新的网格大小并且将旧数据拷贝到新的网格中
        resize(new_rows, new_cols);

        //#pragma omp parallel for collapse(2)
        //插入新的数据
        for (int i = 0; i < static_cast<int>(points.size()); ++i) {
            const auto& point = points[i];
            int row = static_cast<int>((point.first - grid_min_x) / grid_resolution);
            int col = static_cast<int>((point.second - grid_min_y) / grid_resolution);

            if (row >= 0 && row < rows && col >= 0 && col < cols) {
                data_[row * cols + col] = 1;
            }
        }
    }

    // 可视化网格
    void visualize() const {
        std::cout << "Grid visualization:" << std::endl;
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                std::cout << (data_[row * cols + col] ? "X " : ". ");
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // 开始计时
    // 记录开始时间

    // 初始化网格
    int initial_rows = static_cast<int>((grid_max_x - grid_min_x) / grid_resolution) + 1;
    int initial_cols = static_cast<int>((grid_max_y - grid_min_y) / grid_resolution) + 1;
    Grid grid(initial_rows, initial_cols);

    grid.visualize(); // 显示初始网格

    std::vector<std::pair<double, double>> points = {
        {0.0, 0.0}, {0.5, 0.5}, {1.0, 1.0},
        {-1.0, -1.0}, {-0.5, -0.5}, {0.2, 0.8},
    };

    grid.setInput(points);
    grid.visualize(); // 显示更新后的网格

    auto end_time = std::chrono::high_resolution_clock::now(); // 结束计时
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Total program took " << duration << " us" << std::endl;

    // int *p = new int;
    // *p = 10;
    // std::cout << "p = " << *p << std::endl;
    // delete p;
    // delete p; // 双重释放错误
    /*
    free(): double free detected in tcache 2
    Aborted (core dumped)
    */
    return 0;
}