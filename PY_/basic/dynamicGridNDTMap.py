import numpy as np
from collections import defaultdict
from typing import Dict, Tuple, Optional, List

class GridMap:
    """
    A dynamic grid map that stores mean XY positions, mean Z values, and Z variance.
    
    Attributes:
        resolution (float): The grid resolution
        grid_dict (Dict[Tuple[int, int], dict]): Stores grid data with keys as grid indices
    """
    
    def __init__(self, resolution: float):
        """
        Initialize a GridMap with the given resolution.
        
        Args:
            resolution: The grid cell size
        """
        self.resolution = resolution
        self.grid_dict: Dict[Tuple[int, int], dict] = {}
    
    def _get_grid_index(self, xy: np.ndarray) -> Tuple[int, int]:
        """
        Convert continuous coordinates to discrete grid indices.
        
        Args:
            xy: Array of XY coordinates (shape: (2,))
            
        Returns:
            Grid indices as a tuple (x_idx, y_idx)
        """
        return tuple((xy // self.resolution).astype(int))
    
    # ATTENTION: simple for-each version
    # def insert(self, xy: np.ndarray, z: np.ndarray) -> None:
    #     """
    #     Insert point cloud data into the grid map.
        
    #     Args:
    #         xy: Array of XY coordinates (shape: (N, 2))
    #         z: Array of Z values (shape: (N,))
    #     """
    #     assert xy.shape[0] == z.shape[0], "Number of points in xy and z must match"
    #     assert xy.shape[1] == 2, "xy array must have two columns"
        
    #     # Group points by grid index
    #     grid_groups = defaultdict(lambda: {'xy': [], 'z': []})
        
    #     for i in range(xy.shape[0]):
    #         grid_idx = self._get_grid_index(xy[i])
    #         grid_groups[grid_idx]['xy'].append(xy[i])
    #         grid_groups[grid_idx]['z'].append(z[i])
        
    #     # Process each grid group
    #     for grid_idx, data in grid_groups.items():
    #         xy_batch = np.array(data['xy'])
    #         z_batch = np.array(data['z'])
    #         n_new = len(z_batch)
            
    #         # Calculate batch statistics
    #         new_sum_x = np.sum(xy_batch[:, 0])
    #         new_sum_y = np.sum(xy_batch[:, 1])
    #         new_sum_z = np.sum(z_batch)
    #         new_sum_z_sq = np.sum(z_batch**2)
    #         new_mean_z = new_sum_z / n_new
    #         new_variance_z = np.var(z_batch) if n_new > 1 else 0.0
            
    #         # Update grid state
    #         if grid_idx not in self.grid_dict:
    #             # Create new grid
    #             self.grid_dict[grid_idx] = {
    #                 'count': n_new,
    #                 'mean_x': new_sum_x / n_new,
    #                 'mean_y': new_sum_y / n_new,
    #                 'mean_z': new_mean_z,
    #                 'variance_z': new_variance_z,
    #                 'sum_z_sq': new_sum_z_sq
    #             }
    #         else:
    #             # Update existing grid
    #             grid = self.grid_dict[grid_idx]
    #             n_old = grid['count']
    #             n_total = n_old + n_new
                
    #             # Update XY means
    #             grid['mean_x'] = (grid['mean_x'] * n_old + new_sum_x) / n_total
    #             grid['mean_y'] = (grid['mean_y'] * n_old + new_sum_y) / n_total
                
    #             # Update Z mean
    #             old_mean_z = grid['mean_z']
    #             grid['mean_z'] = (old_mean_z * n_old + new_sum_z) / n_total
                
    #             # Update variance using combined formula
    #             delta_old = old_mean_z - grid['mean_z']
    #             delta_new = new_mean_z - grid['mean_z']
                
    #             # Update variance
    #             # Var_total = [n_old*(Var_old + (mean_old - mean_total)^2) + n_new*(Var_new + (mean_new - mean_total)^2)] / n_total
    #             grid['variance_z'] = (
    #                 n_old * (grid['variance_z'] + delta_old**2) + 
    #                 n_new * (new_variance_z + delta_new**2)
    #             ) / n_total
                
    #             # Update point count
    #             grid['count'] = n_total

    # ATTENTION: batch vectorized version
    def insert(self, xy: np.ndarray, z: np.ndarray) -> None:
        """
        Insert point cloud data into the grid map using vectorized operations.
        
        Args:
            xy: Array of XY coordinates (shape: (N, 2))
            z: Array of Z values (shape: (N,))
        """
        assert xy.shape[0] == z.shape[0], "Number of points in xy and z must match"
        assert xy.shape[1] == 2, "xy array must have two columns"
        
        # Calculate grid indices for all points
        grid_indices = (xy // self.resolution).astype(int)
        unique_indices, inverse_indices, counts = np.unique(
            grid_indices, axis=0, return_inverse=True, return_counts=True
        )
        
        # Precompute batch statistics using vectorized operations
        sum_x = np.bincount(inverse_indices, weights=xy[:, 0])
        sum_y = np.bincount(inverse_indices, weights=xy[:, 1])
        sum_z = np.bincount(inverse_indices, weights=z)
        sum_z_sq = np.bincount(inverse_indices, weights=z**2)
        
        # Calculate batch means and variances
        new_mean_x = sum_x / counts
        new_mean_y = sum_y / counts
        new_mean_z = sum_z / counts
        new_variance_z = (sum_z_sq / counts) - new_mean_z**2
        
        # Update grid states
        for i, grid_idx in enumerate(unique_indices):
            grid_idx_tuple = tuple(grid_idx)
            n_new = counts[i]
            
            if grid_idx_tuple not in self.grid_dict:
                # Create new grid
                self.grid_dict[grid_idx_tuple] = {
                    'count': n_new,
                    'mean_x': new_mean_x[i],
                    'mean_y': new_mean_y[i],
                    'mean_z': new_mean_z[i],
                    'variance_z': new_variance_z[i],
                    'sum_z_sq': sum_z_sq[i]
                }
            else:
                # Update existing grid
                grid = self.grid_dict[grid_idx_tuple]
                n_old = grid['count']
                n_total = n_old + n_new
                
                # Update XY means
                grid['mean_x'] = (grid['mean_x'] * n_old + sum_x[i]) / n_total
                grid['mean_y'] = (grid['mean_y'] * n_old + sum_y[i]) / n_total
                
                # Update Z mean
                old_mean_z = grid['mean_z']
                grid['mean_z'] = (old_mean_z * n_old + sum_z[i]) / n_total
                
                # Update variance using combined formula
                delta_old = old_mean_z - grid['mean_z']
                delta_new = new_mean_z[i] - grid['mean_z']
                
                grid['variance_z'] = (
                    n_old * (grid['variance_z'] + delta_old**2) + 
                    n_new * (new_variance_z[i] + delta_new**2)
                ) / n_total
                
                # Update point count and sum_z_sq
                grid['count'] = n_total
                grid['sum_z_sq'] += sum_z_sq[i]
    
    def query(self, xy: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Query grid state at the given XY coordinates.
        
        Args:
            xy: XY coordinates (shape: (2,))
            
        Returns:
            Tuple (mean_x, mean_y, mean_z, var_z) or None if grid doesn't exist
        """
        grid_idx = self._get_grid_index(xy)
        grid = self.grid_dict.get(grid_idx)
        
        if grid is None:
            return None
        
        return (grid['mean_x'], grid['mean_y'], grid['mean_z'], grid['variance_z'])
    
    def get_grid_centers(self) -> List[Tuple[float, float]]:
        """
        Get center coordinates of all populated grids.
        
        Returns:
            List of (center_x, center_y) coordinates
        """
        centers = []
        for idx in self.grid_dict:
            center_x = (idx[0] + 0.5) * self.resolution
            center_y = (idx[1] + 0.5) * self.resolution
            centers.append((center_x, center_y))
        return centers
    
    def get_grid_stats(self) -> Dict[Tuple[int, int], dict]:
        """
        Get statistics for all populated grids.
        
        Returns:
            Dictionary with grid indices as keys and stats as values
        """
        stats = {}
        for idx, grid in self.grid_dict.items():
            stats[idx] = {
                'mean_xy': (grid['mean_x'], grid['mean_y']),
                'mean_z': grid['mean_z'],
                'variance_z': grid['variance_z'],
                'count': grid['count']
            }
        return stats
    
    def get_all_grid_points(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all grid points with their statistics.
        
        Returns:
            Tuple (grid_points, z_means, variances)
            - grid_points: Array of XY coordinates (shape: (N, 2))
            - z_means: Array of Z means (shape: (N,))
            - variances: Array of Z variances (shape: (N,))
        """
        points = []
        z_means = []
        variances = []
        
        for grid in self.grid_dict.values():
            points.append([grid['mean_x'], grid['mean_y']])
            z_means.append(grid['mean_z'])
            variances.append(grid['variance_z'])
        
        return np.array(points), np.array(z_means), np.array(variances)
    


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib as mpl

def visualize_grid_map(grid_map: GridMap, all_xy: np.ndarray, title: str = "") -> None:
    """
    Visualize the grid map with original points and grid statistics.
    
    Args:
        grid_map: The GridMap instance to visualize
        all_xy: Array of all original XY coordinates
        title: Title for the visualization
    """
    # Get grid statistics
    grid_points, z_means, variances = grid_map.get_all_grid_points()
    grid_stats = grid_map.get_grid_stats()
    
    # Calculate grid counts for size scaling
    grid_counts = [stats['count'] for stats in grid_stats.values()]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot original points
    plt.scatter(all_xy[:, 0], all_xy[:, 1], c='red', alpha=1.0, 
                s=15, label='Original Points')
    
    # Plot grid points if any exist
    if len(grid_points) > 0:
        # Create color mapping for Z means
        norm = Normalize(vmin=np.min(z_means), vmax=np.max(z_means))
        cmap = plt.get_cmap('viridis')
        colors = cmap(norm(z_means))
        
        # Calculate point sizes based on point count in each grid
        max_count = np.max(grid_counts) if len(grid_counts) > 0 else 1
        min_size, max_size = 50, 300
        sizes = min_size + (max_size - min_size) * (np.array(grid_counts) / max_count)
        
        # Plot grid points
        scatter = plt.scatter(
            grid_points[:, 0], grid_points[:, 1], 
            c=colors, s=sizes, alpha=0.8, 
            edgecolors='black', linewidth=0.8, 
            label='Grid Centers'
        )
        
        # Add color bar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Z Mean', fontsize=12)
    
    # Draw grid lines
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    resolution = grid_map.resolution
    
    # Generate grid lines
    x_ticks = np.arange(x_min, x_max + resolution, resolution)
    y_ticks = np.arange(y_min, y_max + resolution, resolution)
    
    # Draw vertical lines
    for x in x_ticks:
        plt.axvline(x=x, color='gray', linestyle='-', alpha=0.8, linewidth=0.5)
    
    # Draw horizontal lines
    for y in y_ticks:
        plt.axhline(y=y, color='gray', linestyle='-', alpha=0.8, linewidth=0.5)
    
    # Set plot properties
    plt.title(f'Grid Map Visualization (Resolution={resolution}) {title}', fontsize=14)
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(False)
    plt.legend(loc='upper right')
    
    # Add info box
    if len(grid_points) > 0:
        info_text = f"Total Points: {all_xy.shape[0]}\n" \
                    f"Grid Cells: {len(grid_points)}\n" \
                    f"Z Mean Range: {np.min(z_means):.2f} - {np.max(z_means):.2f}\n" \
                    f"Max Variance: {np.max(variances):.2f}\n" \
                    f"Max Grid Count: {np.max(grid_counts)}"
    else:
        info_text = "No grid data available"
        
    plt.annotate(info_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                 fc="white", ec="gray", alpha=0.8),
                 verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

def generate_point_cloud(num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a point cloud in the range [-5, 5].
    
    Args:
        num_points: Number of points to generate
        
    Returns:
        Tuple (xy, z) where:
        - xy: Array of XY coordinates (shape: (N, 2))
        - z: Array of Z values (shape: (N,))
    """
    xy = np.random.uniform(-5, 5, size=(num_points, 2))
    dist_from_center = np.linalg.norm(xy, axis=1)
    z = 10 * np.exp(-0.2 * dist_from_center) + np.random.normal(0, 0.5, num_points)
    return xy, z

def main():
    """Main demonstration function."""
    # Initialize grid map
    grid_map = GridMap(resolution=0.5)
    
    # Store all original points externally
    all_xy = np.empty((0, 2))
    
    # Generate and insert multiple batches
    np.random.seed(42)
    num_batches = 7
    points_per_batch = 400
    
    print("Grid Map Demonstration")
    print(f"Performing {num_batches} insertions with {points_per_batch} points each")
    print("=" * 60)
    
    for batch in range(num_batches):
        # Generate new point cloud
        xy_batch, z_batch = generate_point_cloud(points_per_batch)
        
        # Store original points
        all_xy = np.vstack((all_xy, xy_batch))
        
        # Insert into grid map
        grid_map.insert(xy_batch, z_batch)
        
        # Visualize current state
        print(f"Insertion #{batch+1} completed")
        print(f"Total points: {all_xy.shape[0]}")
        print(f"Grid cells: {len(grid_map.grid_dict)}")
        
        visualize_grid_map(grid_map, all_xy, title=f"- Batch {batch+1}/{num_batches}")
        
        print(f"Visualization for batch #{batch+1} completed")
        print("-" * 60)
    
    print("All insertions completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()