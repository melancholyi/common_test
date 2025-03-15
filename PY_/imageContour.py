'''
Author: chasey melancholycy@gmail.com
Date: 2025-02-13 13:27:06
FilePath: /mesh_planner/test/python/imageContour.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import cv2
import numpy as np
import time  # Import the time module

# 1. 读取本地图片
image_path = './image.jpg'  # 替换为你的图片路径
image = cv2.imread(image_path)

if image is None:
    print("无法加载图片，请检查路径是否正确！")
    exit()

# 1.5. 调整图像尺寸为原来的一半
height, width = image.shape[:2]
new_size = (width // 2, height // 2)
print("new_size:", new_size)  
resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)  # 使用线性插值[^1^]



# 2. 预处理：高斯模糊
blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # 使用5x5的高斯核进行模糊


start_time = time.time()  # Start timing


# 3. 将图片转换为灰度图
gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)

# 4. 使用膨胀操作增强边缘
kernel = np.ones((3, 3), np.uint8)  # 定义膨胀的核
dilated_image = cv2.dilate(gray_image, kernel, iterations=1)  # 进行一次膨胀

# 5. 使用Canny边缘检测器提取轮廓
edges = cv2.Canny(dilated_image, 100, 200)

# 6. 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 7. 对每个轮廓拟合最小椭圆
for contour in contours:
    # 忽略太小的轮廓
    if len(contour) > 5:  # 椭圆拟合至少需要5个点
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(resized_image, ellipse, (0, 255, 0), 2)  # 绘制椭圆，颜色为绿色

end_time = time.time()
print(f"ellipsoid extract runtime: {(end_time - start_time)*1000:.6f} ms")

# 8. 显示结果
cv2.imshow('Original Image', resized_image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Edges', edges)
cv2.imshow('Contours with Fitted Ellipses', resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()