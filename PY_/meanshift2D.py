import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# 1. 读取图片并转换为灰度图像
image_path = './image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
if image is None:
    raise ValueError("无法加载图片，请检查路径是否正确！")

# 将图片转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 将灰度图像转换为一维数组，形状为 (height * width, 1)
height, width = gray_image.shape
image_reshaped = gray_image.reshape(-1, 1)

# 2. 使用 MeanShift 聚类
# 估计带宽
bandwidth = estimate_bandwidth(image_reshaped, quantile=0.2, n_samples=500)
print(f"Estimated bandwidth: {bandwidth}")

# 创建 MeanShift 模型
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(image_reshaped)

# 获取聚类结果
labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters = len(cluster_centers)
print(f"Estimated number of clusters: {n_clusters}")

# 3. 将聚类结果可视化
# 将每个像素替换为聚类中心的灰度值
segmented_image = cluster_centers[labels].reshape(height, width).astype(np.uint8)

# 4. 计算每个聚类中心的位置
# 获取每个像素的坐标
y_coords, x_coords = np.indices((height, width))
coords = np.stack((y_coords.ravel(), x_coords.ravel()), axis=1)

# 计算每个聚类的中心位置
cluster_centers_coords = []
for cluster_id in range(n_clusters):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_coords = coords[cluster_indices]
    center_y, center_x = np.mean(cluster_coords, axis=0).astype(int)
    cluster_centers_coords.append((center_x, center_y))

# 5. 在灰度图像和聚类结果上绘制红色圆圈标记聚类中心
marked_gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制红色圆圈
marked_segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制红色圆圈

for center in cluster_centers_coords:
    cv2.circle(marked_gray_image, center, radius=10, color=(0, 0, 255), thickness=2)  # 红色圆圈
    cv2.circle(marked_segmented_image, center, radius=10, color=(0, 0, 255), thickness=2)  # 红色圆圈

# 6. 显示结果
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Gray Image")
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Gray Image with Cluster Centers")
plt.imshow(marked_gray_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title(f"MeanShift Segmentation with Cluster Centers ({n_clusters} clusters)")
plt.imshow(marked_segmented_image)
plt.axis('off')

plt.show()