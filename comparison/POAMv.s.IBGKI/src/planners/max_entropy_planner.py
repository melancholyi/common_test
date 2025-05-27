'''
Author: chasey && melancholycy@gmail.com
Date: 2025-04-09 01:34:44
LastEditTime: 2025-05-19 05:10:04
FilePath: /POAM/src/planners/max_entropy_planner.py
Description: 
Reference: 
Copyright (c) 2025 by chasey && melancholycy@gmail.com, All Rights Reserved. 
'''
from typing import List

import numpy as np

from . import BasePlanner


class MaxEntropyPlanner(BasePlanner):
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        num_candidates: int = 1000,
    ) -> None:
        super().__init__(task_extent, rng)
        self.num_candidates = num_candidates

    def plan(self, model, robot_location: np.ndarray, *args, **kwargs) -> np.ndarray:
        assert model is not None, "Model must be provided."
        xmin, xmax, ymin, ymax = self.task_extent
        x = self.rng.uniform(xmin, xmax, self.num_candidates)
        y = self.rng.uniform(ymin, ymax, self.num_candidates)
        candidates = np.column_stack((x, y))
        # TODO: make sure model return std rather than var
        _, std = model.predict(candidates)
        std = std.ravel()
        entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
        diffs = candidates - robot_location.reshape(1, 2)
        dists = np.hypot(diffs[:, 0], diffs[:, 1])
        normalized_entropy = (entropy - entropy.min()) / entropy.ptp()
        normalized_dists = (dists - dists.min()) / dists.ptp()
        scores = normalized_entropy - 0.5 * normalized_dists
        sorted_indices = np.argsort(scores)
        goal = candidates[sorted_indices[-1]]
        return np.atleast_2d(goal) # (1, 2)
