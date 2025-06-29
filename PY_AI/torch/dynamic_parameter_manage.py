import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class DynamicHyperparamOptimizer(nn.Module):
    def __init__(self, base_model, resolution=1, init_val=0.0):
        """
        base_model: 基础神经网络模型
        resolution: 分辨率 - 可以是标量(所有维度相同)或元组(每个维度独立)
        init_val: 新超参数的初始值
        """
        super().__init__()
        self.base_model = base_model
        self.resolution = resolution
        self.init_val = init_val
        
        # 存储动态超参数
        self.hyperparams = nn.ParameterDict()
        
        # 多维索引到参数的映射 {index_tuple: parameter}
        self.index_to_param = {}
        
        # 参数创建计数器
        self.param_counter = 0
        
        # 记录创建历史
        self.creation_history = []
        
        # 维度数 (初始化时未知)
        self.dimensions = None

    def _normalize_resolution(self, position):
        """根据位置维度调整分辨率格式"""
        if not isinstance(position, tuple):
            position = (position,)
        
        # 首次调用时确定维度数
        if self.dimensions is None:
            self.dimensions = len(position)
            
            # 标准化分辨率格式
            if isinstance(self.resolution, (int, float)):
                self.resolution = (self.resolution,) * self.dimensions
            elif len(self.resolution) == 1:
                self.resolution = self.resolution * self.dimensions
            elif len(self.resolution) != self.dimensions:
                raise ValueError(f"分辨率维度({len(self.resolution)})与位置维度({self.dimensions})不匹配")
        
        return position

    def get_hyperparam(self, position):
        """
        根据多维位置获取或创建超参数
        position: 位置坐标 (标量, 元组, 或列表)
        """
        # 标准化位置格式和分辨率
        position = self._normalize_resolution(position)
        
        # 计算归一化索引
        index_tuple = tuple(int(p // res) for p, res in zip(position, self.resolution))
        
        # 检查是否已存在
        if index_tuple in self.index_to_param:
            return self.index_to_param[index_tuple]
        
        # 创建新参数
        param_name = f"hyper_{self.param_counter}"
        new_param = nn.Parameter(torch.tensor(float(self.init_val)))
        
        # 注册并保存
        self.hyperparams[param_name] = new_param
        self.index_to_param[index_tuple] = new_param
        self.param_counter += 1
        
        # 记录创建信息
        creation_info = {
            "name": param_name,
            "index": index_tuple,
            "position": position,
            "initial_value": self.init_val,
            "resolution": self.resolution
        }
        self.creation_history.append(creation_info)
        
        # 打印详细信息
        dim_str = ", ".join(str(p) for p in position)
        idx_str = ", ".join(str(i) for i in index_tuple)
        res_str = ", ".join(str(r) for r in self.resolution)
        
        print(f"✨ 创建 {len(position)}D 超参数: {param_name}")
        print(f"  位置: ({dim_str}) → 索引: ({idx_str})")
        print(f"  分辨率: ({res_str})")
        print(f"  初始值: {self.init_val}")
        print(f"  当前总超参数数: {len(self.hyperparams)}")
        
        return new_param

    def print_creation_history(self):
        """打印所有参数创建历史"""
        if not self.creation_history:
            print("尚未创建任何超参数")
            return
            
        print("\n" + "="*70)
        print(f"{self.dimensions}D 超参数创建历史:")
        print("="*70)
        for i, info in enumerate(self.creation_history):
            pos_str = ", ".join(f"{p:.2f}" for p in info["position"])
            idx_str = ", ".join(str(i) for i in info["index"])
            current_val = self.index_to_param[info["index"]].item()
            
            print(f"{i+1}. {info['name']}:")
            print(f"  位置: [{pos_str}] | 索引: [{idx_str}]")
            print(f"  初始值: {info['initial_value']:.4f} | 当前值: {current_val:.4f}")
        print("="*70 + "\n")

    def visualize_2d_grid(self, x_range, y_range, ax=None):
        """可视化二维参数网格"""
        if self.dimensions != 2:
            print(f"可视化仅支持2D数据，当前维度: {self.dimensions}D")
            return
            
        min_x, max_x = x_range
        min_y, max_y = y_range
        
        # 计算索引范围
        x_res, y_res = self.resolution
        x_start = int(min_x // x_res)
        x_end = int(max_x // x_res) + 1
        y_start = int(min_y // y_res)
        y_end = int(max_y // y_res) + 1
        
        # 创建网格数据
        grid = np.full((x_end - x_start, y_end - y_start), np.nan)
        
        # 填充网格值
        for i, x_idx in enumerate(range(x_start, x_end)):
            for j, y_idx in enumerate(range(y_start, y_end)):
                index = (x_idx, y_idx)
                if index in self.index_to_param:
                    grid[i, j] = self.index_to_param[index].item()
        
        # 创建可视化
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure
        
        # 创建伪彩色图
        cax = ax.matshow(grid, cmap='viridis', origin='lower', 
                         extent=[y_start, y_end, x_start, x_end])
        
        # 添加颜色条
        fig.colorbar(cax, ax=ax, label='Hyperparameter Value')
        
        # 设置坐标轴
        ax.set_xlabel('Y Index')
        ax.set_ylabel('X Index')
        ax.set_title(f'2D Hyperparameter Grid\nResolution: X={x_res}, Y={y_res}')
        
        # 添加网格线和文本标签
        ax.grid(True, linestyle='--', alpha=0.7)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j + 0.5, i + 0.5, f'{grid[i, j]:.2f}', 
                            ha='center', va='center', color='white')
        
        return fig

    def forward(self, x, positions):
        """
        x: 输入数据 [batch_size, ...]
        positions: 每个样本对应的多维位置 [pos1, pos2, ...]
                   pos可以是标量(1D)或元组/列表(多维)
        """
        # 获取基础模型输出
        base_output = self.base_model(x)
        
        # 为每个样本获取对应的超参数
        sample_hyperparams = []
        for pos in positions:
            # 转换为可处理格式
            if isinstance(pos, torch.Tensor):
                pos = pos.tolist()
            elif isinstance(pos, (int, float)):
                pos = (pos,)
                
            hp = self.get_hyperparam(tuple(pos))
            sample_hyperparams.append(hp)
        
        hyper_tensor = torch.stack(sample_hyperparams)
        
        # 示例损失计算 (替换为实际需求)
        # 这里将超参数作为权重与模型输出结合
        weighted_output = base_output * hyper_tensor.unsqueeze(1)
        loss = weighted_output.mean()
        
        # 添加L2正则化
        reg_loss = sum(hp**2 for hp in self.hyperparams.values()) * 0.01
        total_loss = loss + reg_loss
        
        return total_loss

# 1. 创建基础模型
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # 添加sigmoid使输出在0-1范围

# 示例使用 - 二维位置场景
if __name__ == "__main__":
    print("="*70)
    print("多维动态超参数优化系统演示")
    print("="*70)
    

    
    base_model = BaseModel()
    
    # 2. 初始化动态优化器 - 使用二维分辨率
    print("\n初始化优化器: 2D分辨率 (3, 2)")
    optimizer = DynamicHyperparamOptimizer(
        base_model=base_model,
        resolution=(3, 2),  # x方向分辨率3, y方向分辨率2
        init_val=0.5
    )
    
    # 3. 创建组合优化器
    all_params = [
        {'params': base_model.parameters()},
        {'params': optimizer.hyperparams.parameters()}
    ]
    adam = optim.Adam(all_params, lr=0.05)
    
    # 4. 模拟训练数据 - 二维位置
    # 每个样本的位置坐标 (x, y)
    positions_list = [
        [(1.2, 1.5), (2.8, 3.1), (4.1, 1.9)],  # 第一组位置
        [(1.0, 1.0), (5.2, 2.1), (7.5, 3.8)],   # 第二组位置
        [(3.2, 4.0), (6.1, 1.0), (9.0, 2.5)],    # 第三组位置
        [(0.5, 0.5), (3.1, 5.2), (8.4, 4.1)]     # 第四组位置
    ]
    
    # 训练循环
    for epoch, positions in enumerate(positions_list):
        # 模拟输入数据 (batch_size=3, feature_size=10)
        data = torch.randn(3, 10)
        
        # 前向传播
        adam.zero_grad()
        loss = optimizer(data, positions)
        
        # 反向传播
        loss.backward()
        adam.step()
        
        print(f"\nEpoch {epoch+1} | Loss: {loss.item():.4f}")
        print(f"当前位置: {positions}")
        
        # 打印当前参数值
        print("当前超参数值:")
        for idx, param in optimizer.index_to_param.items():
            print(f"  索引 {idx}: {param.item():.4f}")
    
    # 打印完整创建历史
    optimizer.print_creation_history()
    
    # 可视化二维参数网格
    fig = optimizer.visualize_2d_grid((0, 10), (0, 6))
    plt.savefig('2d_hyperparameter_grid.png', dpi=300, bbox_inches='tight')
    print("已保存二维参数网格图像: 2d_hyperparameter_grid.png")
    
    # # 三维演示
    # print("\n\n" + "="*70)
    # print("三维动态超参数优化演示")
    # print("="*70)
    
    # # 创建新优化器 - 三维分辨率
    # print("\n初始化优化器: 3D分辨率 (2, 3, 4)")
    # optimizer_3d = DynamicHyperparamOptimizer(
    #     base_model=BaseModel(),
    #     resolution=(2, 3, 4),
    #     init_val=0.3
    # )
    
    # # 三维位置示例
    # positions_3d = [
    #     (1.5, 2.5, 3.5),
    #     (3.0, 1.0, 7.0),
    #     (0.5, 4.5, 2.5),
    #     (5.0, 2.0, 6.0)
    # ]
    
    # # 处理三维位置
    # data = torch.randn(4, 10)
    # loss = optimizer_3d(data, positions_3d)
    # print(f"\n处理三维位置后损失: {loss.item():.4f}")
    
    # # 打印三维创建历史
    # optimizer_3d.print_creation_history()
    
    # # 一维演示
    # print("\n\n" + "="*70)
    # print("一维动态超参数优化演示")
    # print("="*70)
    
    # # 创建新优化器 - 一维分辨率
    # print("\n初始化优化器: 1D分辨率 (2.5)")
    # optimizer_1d = DynamicHyperparamOptimizer(
    #     base_model=BaseModel(),
    #     resolution=2.5,
    #     init_val=0.4
    # )
    
    # # 一维位置示例
    # positions_1d = [1.2, 3.7, 2.8, 6.3, 8.9]
    
    # # 处理一维位置
    # data = torch.randn(5, 10)
    # loss = optimizer_1d(data, positions_1d)
    # print(f"\n处理一维位置后损失: {loss.item():.4f}")
    
    # # 打印一维创建历史
    # optimizer_1d.print_creation_history()
    
    # # 打印所有参数值
    # print("\n一维超参数值:")
    # for idx, param in optimizer_1d.index_to_param.items():
    #     print(f"  索引 {idx}: {param.item():.4f}")