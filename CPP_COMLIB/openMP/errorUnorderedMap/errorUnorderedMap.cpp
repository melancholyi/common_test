#include <unordered_map>
#include <omp.h>
#include <string>
#include <utility>

// 自定义哈希函数（简化版）
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
    }
};

std::unordered_map<std::pair<int, int>, std::string, PairHash> shared_map;

int main() {
    // 设置初始小容量以强制频繁rehash
    shared_map.reserve(1);

    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < 10000; ++i) {
        // 生成可能重复的键
        auto key = std::make_pair(i % 100, i / 100);
        std::string value = "val_" + std::to_string(i);
        
        // 故意不加锁的并发插入
        shared_map[key] = value;  // 多线程触发rehash
        /*
        double free or corruption (!prev)
        Aborted (core dumped)
        */
    }

    return 0;
}