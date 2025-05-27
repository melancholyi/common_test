'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-09 01:34:44
LastEditTime: 2025-05-18 15:29:02
FilePath: /POAM/src/sensors/base_sensor.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class BaseSensor(metaclass=ABCMeta):
    def __init__(
        self,
        matrix: np.ndarray,
        env_extent: List[float],
        rate: float,
        noise_scale: float,
        rng: np.random.RandomState,
    ) -> None:
        self.matrix = matrix # map elevation, 2D array([W, H]) 
        self.env_extent = env_extent # map range: [x_min, x_max, y_min, y_max]
        self.num_rows, self.num_cols = matrix.shape
        self.x_cell_size = (env_extent[1] - env_extent[0]) / self.num_cols # x cell size, resolution_x
        self.y_cell_size = (env_extent[3] - env_extent[2]) / self.num_rows # y cell size, resolution_y
        self.dt = 1.0 / rate # sample delta time
        self.noise_scale = noise_scale # noise scale
        self.rng = rng # random number generator

    @abstractmethod
    def sense(self, states: np.ndarray) -> np.ndarray: # virtual function, waiting to be implemented by derived classes
        raise NotImplementedError

    def xs_to_cols(self, xs: np.ndarray) -> np.ndarray:
        cols = ((xs - self.env_extent[0]) / self.x_cell_size).astype(int) # x_value to column index
        np.clip(cols, 0, self.num_cols - 1, out=cols)
        return cols

    def ys_to_rows(self, ys: np.ndarray) -> np.ndarray:
        rows = ((ys - self.env_extent[2]) / self.y_cell_size).astype(int)
        np.clip(rows, 0, self.num_rows - 1, out=rows)
        return rows

    def get(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        values = self.matrix[rows, cols]
        return values

    def set(self, xs: np.ndarray, ys: np.ndarray, values: np.ndarray) -> None:
        cols = self.xs_to_cols(xs)
        rows = self.ys_to_rows(ys)
        self.matrix[rows, cols] = values
