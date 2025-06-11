import numpy as np
import matplotlib.pyplot as plt

class VoxelGridDownsampling2D:
    def __init__(self, voxel_size):
        # 初始化体素网格的大小
        self.voxel_size = voxel_size

    def downsample(self, point_cloud):
        """
        对二维点云进行降采样

        参数：
        point_cloud -- 输入的二维点云数组，形状为 (n, 2)，其中 n 是点的数量

        返回：
        downsampled_point_cloud -- 降采样后的二维点云数组
        """
        if len(point_cloud) == 0:
            return np.array([])

        # 获取点云的 x 和 y 坐标
        x_coords = point_cloud[:, 0]
        y_coords = point_cloud[:, 1]

        # 计算每个点所属的体素网格索引
        voxel_x_indices = np.floor(x_coords / self.voxel_size).astype(int)
        voxel_y_indices = np.floor(y_coords / self.voxel_size).astype(int)

        # 将体素网格索引合并为一个唯一键
        voxel_keys = voxel_x_indices * (1 << 32) + voxel_y_indices

        # 找到每个体素网格的唯一键及其对应的索引
        unique_voxel_keys, unique_indices = np.unique(voxel_keys, return_index=True)

        # 使用唯一索引从原始点云中选择点
        downsampled_point_cloud = point_cloud[unique_indices]

        return downsampled_point_cloud

# 生成更多的点
np.random.seed(42)  # 设置随机种子以确保结果可重复
num_points = 1000
point_cloud = np.random.rand(num_points, 2) * 10  # 生成 1000 个点，范围在 [0, 10) 的二维点云

# 创建降采样对象，体素大小为 0.5
downsampler = VoxelGridDownsampling2D(voxel_size=0.5)

# 对点云进行降采样
downsampled_pc = downsampler.downsample(point_cloud)

# 可视化原始点和降采样后的点
plt.figure(figsize=(10, 10))

# 绘制原始点（绿色）
plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='green', s=10, alpha=0.5)

# 绘制降采样后的点（红色）
plt.scatter(downsampled_pc[:, 0], downsampled_pc[:, 1], c='red', s=25, edgecolors='k')

# 设置网格大小与分辨率一致
plt.grid(True)
plt.xticks(np.arange(0, 10.1, 0.5))
plt.yticks(np.arange(0, 10.1, 0.5))

plt.axis('equal')
plt.show()