import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# 1. 读取图片并转换为灰度图像
image_path = './image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)
if image is None:
    raise ValueError("无法加载图片，请检查路径是否正确！")

# 将图片转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray_image.shape

# 2. 准备数据：将每个像素的灰度值和空间位置作为特征
# 获取每个像素的坐标
y_coords, x_coords = np.indices((height, width))
coords = np.stack((x_coords.ravel(), y_coords.ravel()), axis=1)
gray_values = gray_image.ravel().reshape(-1, 1)
data = np.hstack((coords, gray_values))

# 3. 使用 DBSCAN 聚类
# 设置 DBSCAN 的参数
eps = 100  # 邻域半径，可以根据图像大小调整
min_samples = 100  # 最小样本数，可以根据图像密度调整
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(data)

# 获取聚类数量（忽略噪声点，标签为 -1）
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Estimated number of clusters: {n_clusters}")

# 4. 可视化聚类结果
# 创建一个与原始图像大小相同的聚类结果图像
segmented_image = np.zeros((height, width), dtype=np.uint8)
for label in set(labels):
    if label == -1:  # 噪声点
        continue
    cluster_indices = np.where(labels == label)[0]
    segmented_image[data[cluster_indices, 1].astype(int), data[cluster_indices, 0].astype(int)] = label + 1

# 5. 计算每个聚类中心的位置
cluster_centers_coords = []
for label in set(labels):
    if label == -1:  # 噪声点
        continue
    cluster_indices = np.where(labels == label)[0]
    cluster_coords = data[cluster_indices, :2]
    center_x, center_y = np.mean(cluster_coords, axis=0).astype(int)
    cluster_centers_coords.append((center_x, center_y))

# 6. 在灰度图像和聚类结果上绘制红色圆圈标记聚类中心
marked_gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制红色圆圈
marked_segmented_image = cv2.cvtColor(segmented_image * 30, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以便绘制红色圆圈

for center in cluster_centers_coords:
    cv2.circle(marked_gray_image, center, radius=10, color=(0, 0, 255), thickness=2)  # 红色圆圈
    cv2.circle(marked_segmented_image, center, radius=10, color=(0, 0, 255), thickness=2)  # 红色圆圈

# 7. 显示结果
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
plt.title(f"DBSCAN Segmentation with Cluster Centers ({n_clusters} clusters)")
plt.imshow(marked_segmented_image)
plt.axis('off')

plt.show()