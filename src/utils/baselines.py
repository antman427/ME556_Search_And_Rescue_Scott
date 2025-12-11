"""
Simple greedy controller that moves toward highest probability regions.
Use this as a baseline or fallback if ergodic control is too slow.
"""

import numpy as np
from typing import Tuple


class GreedyController:
    """
    Greedy controller that always moves toward the highest probability region.
    Much faster than ergodic control, good for baseline comparison.
    """
    
    def __init__(self, map_size: Tuple[float, float]):
        """
        Initialize greedy controller.
        
        Args:
            map_size: Map size (width, height) in meters
        """
        self.map_size = np.array(map_size)
        
        # Control parameters
        self.target_velocity = 0.6  # m/s
        self.k_angular = 2.5  # Angular gain
        
    def get_next_control(self, current_state: np.ndarray, 
                        target_dist: np.ndarray) -> np.ndarray:
        """
        Get next control action - move toward highest probability.
        
        Args:
            current_state: Current robot state [x, y, theta]
            target_dist: Target distribution
            
        Returns:
            Control [v, omega]
        """
        # Find highest probability location
        h, w = target_dist.shape
        max_idx = np.unravel_index(np.argmax(target_dist), target_dist.shape)
        
        # Convert to world coordinates
        target_x = (max_idx[0] + 0.5) * self.map_size[0] / h
        target_y = (max_idx[1] + 0.5) * self.map_size[1] / w
        target_pos = np.array([target_x, target_y])
        
        # Vector to target
        to_target = target_pos - current_state[:2]
        distance = np.linalg.norm(to_target)
        
        # Desired heading
        desired_heading = np.arctan2(to_target[1], to_target[0])
        
        # Heading error
        heading_error = np.arctan2(
            np.sin(desired_heading - current_state[2]),
            np.cos(desired_heading - current_state[2])
        )
        
        # Control law
        if abs(heading_error) > np.pi / 4:
            # Large heading error - turn more, go slower
            v = 0.3
            omega = self.k_angular * heading_error
        else:
            # Small heading error - go faster
            v = self.target_velocity
            omega = self.k_angular * heading_error * 0.5
        
        return np.array([v, omega])


class FrontierController:
    """
    Frontier-based exploration controller.
    Moves toward the nearest frontier (boundary between known and unknown space).
    """
    
    def __init__(self, map_size: Tuple[float, float], resolution: float = 0.1):
        """
        Initialize frontier controller.
        
        Args:
            map_size: Map size in meters
            resolution: Grid resolution
        """
        self.map_size = np.array(map_size)
        self.resolution = resolution
        
        self.target_velocity = 0.6  # m/s
        self.k_angular = 2.5
        
    def get_next_control(self, current_state: np.ndarray,
                        occupancy_map: np.ndarray,
                        explored_mask: np.ndarray) -> np.ndarray:
        """
        Get control to move toward nearest frontier.
        
        Args:
            current_state: [x, y, theta]
            occupancy_map: Occupancy grid
            explored_mask: Boolean mask of explored cells
            
        Returns:
            Control [v, omega]
        """
        # Find frontiers (explored free cells adjacent to unexplored)
        from scipy.ndimage import binary_dilation
        
        # Dilate explored region
        dilated = binary_dilation(explored_mask)
        
        # Frontiers are newly covered cells that are free
        frontiers = dilated & ~explored_mask & (occupancy_map < 0.5)
        
        if not np.any(frontiers):
            # No frontiers - just move forward
            return np.array([self.target_velocity, 0.0])
        
        # Find nearest frontier
        frontier_indices = np.argwhere(frontiers)
        robot_grid = (
            int(current_state[0] / self.resolution),
            int(current_state[1] / self.resolution)
        )
        
        distances = np.linalg.norm(frontier_indices - np.array(robot_grid), axis=1)
        nearest_idx = frontier_indices[np.argmin(distances)]
        
        # Convert to world coordinates
        target_pos = np.array([
            (nearest_idx[0] + 0.5) * self.resolution,
            (nearest_idx[1] + 0.5) * self.resolution
        ])
        
        # Move toward target
        to_target = target_pos - current_state[:2]
        desired_heading = np.arctan2(to_target[1], to_target[0])
        
        heading_error = np.arctan2(
            np.sin(desired_heading - current_state[2]),
            np.cos(desired_heading - current_state[2])
        )
        
        if abs(heading_error) > np.pi / 4:
            v = 0.3
            omega = self.k_angular * heading_error
        else:
            v = self.target_velocity
            omega = self.k_angular * heading_error * 0.5
        
        return np.array([v, omega])


class RandomWalkController:
    """
    Random walk controller for baseline comparison.
    """
    
    def __init__(self):
        """Initialize random walk controller."""
        self.velocity = 0.5
        self.change_direction_prob = 0.1
        self.current_omega = 0.0
        
    def get_next_control(self, current_state: np.ndarray,
                        target_dist: np.ndarray = None) -> np.ndarray:
        """
        Get random control action.
        
        Args:
            current_state: Current state (unused)
            target_dist: Target distribution (unused)
            
        Returns:
            Control [v, omega]
        """
        # Randomly change direction
        if np.random.random() < self.change_direction_prob:
            self.current_omega = np.random.uniform(-1.0, 1.0)
        
        return np.array([self.velocity, self.current_omega])
