'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-09 01:34:44
LastEditTime: 2025-05-18 13:39:17
FilePath: /POAM/data/visualize.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
import numpy as np
import pyvista as pv
import src

env_extent = [-31.0, 31.0, -31.0, 31.0]
image_path = f"/home/Workspace/REF/ML/kernel/POAM/images"
# camera
camera_position = (30, -82.6, 116.2)
camera_focal_point = (-4.25, 4.33, 2.12)
camera_up = (-0.07, 0.78, 0.63)

def visualize_env(env: np.ndarray, save_path: str = None) -> None:
    plotter = pv.Plotter()
    env = env[:, :, np.newaxis]
    x_range = env_extent[1] - env_extent[0]
    y_range = env_extent[3] - env_extent[2]
    z_range = np.max(env) - np.min(env)
    resolution = [x_range / env.shape[0], y_range / env.shape[1], z_range]
    origin = np.array([env_extent[0], env_extent[2], 0])
    image_data = pv.ImageData(dimensions=env.shape, origin=origin, spacing=resolution)
    image_data.point_data["values"] = env.flatten()
    terrain = image_data.warp_by_scalar()
    plotter.add_mesh(terrain, cmap="jet", lighting=True, show_scalar_bar=False)
    plotter.add_axes()
    plotter.camera.position = camera_position
    plotter.camera.focal_point = camera_focal_point
    plotter.camera.up = camera_up
    plotter.show()
    if save_path is not None:
        plotter.screenshot(save_path)
    plotter.close()

def main() -> None:
    env_names = ["n44w111", "n17e073", "n47w124", "n35w107"]
    scaler = src.scalers.MinMaxScaler(expected_min=0.0, expected_max=25.0) # 规范化到0-25范围内
    for env_name in env_names:
        with np.load(f"./arrays/{env_name}.npz") as data:
            env = data["arr_0"]
        scaler.fit(env.flatten())
        env = scaler.preprocess(env)
        print(f'env.shape:{env.shape}')
        print(f'env[0:10]:{env[0:10]}')
        print(f'env_name: {env_name}')
        visualize_env(env, f"{image_path}/env_{env_name}.png")

if __name__ == "__main__":
    main()
