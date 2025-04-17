import numpy as np
import matplotlib.pyplot as plt
import math

def create_rotated_ellipse_mask(shape, a=2, b=1, theta_deg=30):
    """
    为每个索引 (i, j) 创建一个旋转椭圆掩码。

    参数:
        shape: 张量的形状 (height, width)
        a: 半长轴
        b: 半短轴
        theta_deg: 旋转角度（度）

    返回:
        masks: 形状为 (height, width, 5, 5) 的掩码
    """
    theta_rad = math.radians(theta_deg)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    # 掩码大小为 5x5
    mask_size = 5
    half_size = mask_size // 2

    # 创建掩码
    masks = np.zeros((shape[0], shape[1], mask_size, mask_size), dtype=bool)

    # 生成网格坐标
    y_grid, x_grid = np.indices((mask_size, mask_size)) - half_size

    # 对每个索引 (i, j) 生成掩码
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 计算旋转后的坐标
            x_rot = x_grid * cos_theta + y_grid * sin_theta
            y_rot = y_grid * cos_theta - x_grid * sin_theta

            # 椭圆方程
            mask = (x_rot**2 / a**2) + (y_rot**2 / b**2) <= 1
            masks[i, j] = mask

    return masks

def extract_data(data, masks):
    """
    根据掩码从原始数据中提取值。

    参数:
        data: 原始数据 (height, width)
        masks: 掩码 (height, width, mask_size, mask_size)

    返回:
        extracted_data: 提取的数据 (height, width, mask_size, mask_size)
    """
    height, width = data.shape
    mask_size = masks.shape[2]
    half_size = mask_size // 2

    extracted_data = np.zeros((height, width, mask_size, mask_size))

    for i in range(height):
        for j in range(width):
            # 计算掩码的边界
            i_start = max(0, i - half_size)
            i_end = min(height, i + half_size + 1)
            j_start = max(0, j - half_size)
            j_end = min(width, j + half_size + 1)

            # 提取数据
            data_patch = data[i_start:i_end, j_start:j_end]

            # 如果提取的数据范围小于掩码大小，进行填充
            if data_patch.shape[0] < mask_size or data_patch.shape[1] < mask_size:
                padded_data_patch = np.zeros((mask_size, mask_size))
                padded_data_patch[:data_patch.shape[0], :data_patch.shape[1]] = data_patch
                data_patch = padded_data_patch

            # 应用掩码，将掩码为 False 的位置设为 0
            extracted_patch = np.where(masks[i, j], data_patch, 0)

            # 将处理后的数据块赋值到目标数组
            extracted_data[i, j] = extracted_patch

    return extracted_data

# 示例
shape = (100, 100)
data = np.random.rand(*shape)  # 示例数据

# 创建掩码
masks = create_rotated_ellipse_mask(shape, a=2, b=1, theta_deg=30)

# 提取数据
extracted_data = extract_data(data, masks)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# 显示掩码
ax[0].imshow(masks[2, 2], cmap='gray')
ax[0].set_title("Rotated Ellipse Mask at (2, 2)")

# 显示提取的数据
ax[1].imshow(extracted_data[2, 2], cmap='viridis')
ax[1].set_title("Extracted Data at (2, 2)")

plt.tight_layout()
plt.show()