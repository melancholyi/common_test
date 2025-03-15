'''
Author: chasey melancholycy@gmail.com
Date: 2025-03-14 04:53:03
FilePath: /mesh_planner/test/python/joystickDriver.py
Description: 

Copyright (c) 2025 by chasey (melancholycy@gmail.com), All Rights Reserved. 
'''
import pygame
import sys

# 初始化pygame
pygame.init()

# 初始化手柄
pygame.joystick.init()

# 检查手柄是否连接
if pygame.joystick.get_count() == 0:
    print("未检测到手柄")
    sys.exit()

# 获取第一个手柄
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("手柄名称:", joystick.get_name())
print("手柄轴数:", joystick.get_numaxes())
print("手柄按钮数:", joystick.get_numbuttons())
print("手柄帽数:", joystick.get_numhats())

# 主循环
try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"按钮 {event.button} 被按下")
            elif event.type == pygame.JOYBUTTONUP:
                print(f"按钮 {event.button} 被释放")
            elif event.type == pygame.JOYAXISMOTION:
                print(f"摇杆 {event.axis} 移动到 {event.value}")
            elif event.type == pygame.JOYHATMOTION:
                print(f"方向键 {event.hat} 移动到 {event.value}")

except KeyboardInterrupt:
    print("程序被用户中断")

finally:
    pygame.quit()