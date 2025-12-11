"""
Environment module for simulating 2D search and rescue environments.
Loads arbitrary images as maps and simulates robot sensors.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


class Objective:
    """Represents a search objective (person, pet, object) in the environment."""
    
    def __init__(self, position: np.ndarray, obj_type: str, size: float = 0.3):
        """
        Initialize an objective.
        
        Args:
            position: (x, y) position in meters
            obj_type: Type of objective ('person', 'pet', 'object')
            size: Radius of objective in meters
        """
        self.position = position
        self.obj_type = obj_type
        self.size = size
        self.discovered = False
        self.discovered_time = None


class Environment:
    """
    2D environment for search and rescue simulation.
    Loads arbitrary images as maps and manages objectives.
    """
    
    def __init__(self, map_image_path: Optional[str] = None, 
                 map_size: Tuple[float, float] = (10.0, 10.0),
                 resolution: float = 0.05):
        """
        Initialize environment.
        
        Args:
            map_image_path: Path to map image (None for empty map)
            map_size: Physical size of map in meters (width, height)
            resolution: Grid resolution in meters per pixel
        """
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = (int(map_size[0] / resolution), 
                         int(map_size[1] / resolution))
        
        # Load or create map
        if map_image_path:
            self.occupancy_map = self._load_map(map_image_path)
        else:
            self.occupancy_map = np.ones(self.grid_size, dtype=np.float32)
        
        # Objectives in the environment
        self.objectives: List[Objective] = []
        
        # Current time
        self.current_time = 0.0
        
    def _load_map(self, image_path: str) -> np.ndarray:
        """
        Load environment map from image.
        Dark pixels = obstacles (0), Light pixels = free space (1)
        
        Args:
            image_path: Path to map image
            
        Returns:
            Binary occupancy map (0=occupied, 1=free)
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load map image: {image_path}")
        
        # Resize to match grid size
        img = cv2.resize(img, self.grid_size)
        
        # Threshold: dark = occupied (0), light = free (1)
        occupancy = (img > 128).astype(np.float32)
        
        return occupancy
    
    def add_objective(self, position: np.ndarray, obj_type: str, size: float = 0.3):
        """
        Add an objective to the environment.
        
        Args:
            position: (x, y) position in meters
            obj_type: Type of objective
            size: Size in meters
        """
        obj = Objective(position, obj_type, size)
        self.objectives.append(obj)
        
    def add_random_objectives(self, num_objectives: int, obj_types: List[str]):
        """
        Add random objectives in free space, away from walls.
        
        Args:
            num_objectives: Number of objectives to add
            obj_types: List of objective types to randomly choose from
        """
        # Define safe zone (away from walls)
        margin = 1.0  # meters from walls
        safe_x_min = margin
        safe_x_max = self.map_size[0] - margin
        safe_y_min = margin
        safe_y_max = self.map_size[1] - margin
        
        for _ in range(num_objectives):
            # Try to find random free space (with timeout)
            max_attempts = 100
            for attempt in range(max_attempts):
                x = np.random.uniform(safe_x_min, safe_x_max)
                y = np.random.uniform(safe_y_min, safe_y_max)
                
                pos = np.array([x, y])
                
                # Check if position is free and not too close to other objectives
                if self.is_free(pos):
                    # Check distance to other objectives
                    too_close = False
                    for obj in self.objectives:
                        if np.linalg.norm(obj.position - pos) < 2.0:  # At least 2m apart
                            too_close = True
                            break
                    
                    if not too_close:
                        obj_type = np.random.choice(obj_types)
                        self.add_objective(pos, obj_type)
                        break
            
            if attempt == max_attempts - 1:
                # Fallback: place in center if couldn't find good spot
                center_pos = np.array([self.map_size[0] / 2, self.map_size[1] / 2])
                if self.is_free(center_pos):
                    obj_type = np.random.choice(obj_types)
                    self.add_objective(center_pos, obj_type)
    
    def is_free(self, position: np.ndarray) -> bool:
        """
        Check if a position is in free space.
        
        Args:
            position: (x, y) position in meters
            
        Returns:
            True if position is free, False if occupied
        """
        grid_pos = self.world_to_grid(position)
        
        if not self.in_bounds(grid_pos):
            return False
        
        return self.occupancy_map[grid_pos[0], grid_pos[1]] > 0.5
    
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid coordinates (pixels)."""
        grid_x = int(position[0] / self.resolution)
        grid_y = int(position[1] / self.resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid coordinates (pixels) to world coordinates (meters)."""
        x = (grid_pos[0] + 0.5) * self.resolution
        y = (grid_pos[1] + 0.5) * self.resolution
        return np.array([x, y])
    
    def in_bounds(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is within map bounds."""
        return (0 <= grid_pos[0] < self.grid_size[0] and 
                0 <= grid_pos[1] < self.grid_size[1])
    
    def get_lidar_scan(self, position: np.ndarray, num_rays: int = 360,
                       max_range: float = 5.0, noise_std: float = 0.05) -> np.ndarray:
        """
        Simulate LIDAR scan from a position.
        
        Args:
            position: (x, y) position in meters
            num_rays: Number of rays in scan
            max_range: Maximum range in meters
            noise_std: Standard deviation of range noise
            
        Returns:
            Array of ranges for each ray (NaN for no return)
        """
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        ranges = np.full(num_rays, max_range)
        
        for i, angle in enumerate(angles):
            # Ray casting
            direction = np.array([np.cos(angle), np.sin(angle)])
            
            for r in np.linspace(0, max_range, int(max_range / self.resolution)):
                point = position + r * direction
                
                if not self.is_free(point):
                    ranges[i] = r
                    break
        
        # Add noise
        ranges += np.random.normal(0, noise_std, num_rays)
        ranges = np.clip(ranges, 0, max_range)
        
        return ranges
    
    def get_camera_view(self, position: np.ndarray, heading: float,
                       fov: float = np.pi / 3, range_limit: float = 3.0) -> dict:
        """
        Get visible objectives within camera field of view.
        
        Args:
            position: (x, y) position in meters
            heading: Robot heading in radians
            fov: Field of view in radians
            range_limit: Maximum detection range
            
        Returns:
            Dictionary with visible objectives and their relative positions
        """
        visible_objectives = []
        
        for obj in self.objectives:
            # Vector to objective
            to_obj = obj.position - position
            distance = np.linalg.norm(to_obj)
            
            if distance > range_limit:
                continue
            
            # Angle to objective
            angle_to_obj = np.arctan2(to_obj[1], to_obj[0])
            angle_diff = np.arctan2(np.sin(angle_to_obj - heading),
                                   np.cos(angle_to_obj - heading))
            
            # Check if within FOV
            if abs(angle_diff) < fov / 2:
                # Check if line of sight is clear
                if self._has_line_of_sight(position, obj.position):
                    visible_objectives.append({
                        'objective': obj,
                        'distance': distance,
                        'angle': angle_diff,
                        'relative_position': to_obj
                    })
        
        return {'objectives': visible_objectives}
    
    def _has_line_of_sight(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if there's a clear line of sight between two positions."""
        direction = pos2 - pos1
        distance = np.linalg.norm(direction)
        direction = direction / distance
        
        num_checks = int(distance / self.resolution)
        for i in range(num_checks):
            check_pos = pos1 + (i * self.resolution) * direction
            if not self.is_free(check_pos):
                return False
        
        return True
    
    def visualize(self, robot_positions: Optional[List[np.ndarray]] = None,
                  robot_trajectories: Optional[List[List[np.ndarray]]] = None,
                  show_objectives: bool = True,
                  title: str = "Environment"):
        """
        Visualize the environment.
        
        Args:
            robot_positions: Current robot positions
            robot_trajectories: Historical robot trajectories
            show_objectives: Whether to show objectives
            title: Plot title
        """
        plt.figure(figsize=(10, 10))
        
        # Show occupancy map
        plt.imshow(self.occupancy_map.T, cmap='gray', origin='lower',
                  extent=[0, self.map_size[0], 0, self.map_size[1]])
        
        # Show objectives
        if show_objectives:
            for obj in self.objectives:
                color = 'red' if obj.obj_type == 'person' else 'blue'
                marker = 'X' if obj.discovered else 'o'
                plt.plot(obj.position[0], obj.position[1], 
                        marker=marker, markersize=10, color=color,
                        markeredgecolor='white', markeredgewidth=2)
        
        # Show robot trajectories
        if robot_trajectories:
            colors = plt.cm.Set1(np.linspace(0, 1, len(robot_trajectories)))
            for i, traj in enumerate(robot_trajectories):
                if len(traj) > 0:
                    traj_array = np.array(traj)
                    plt.plot(traj_array[:, 0], traj_array[:, 1], 
                            '-', color=colors[i], alpha=0.5, linewidth=2)
        
        # Show current robot positions
        if robot_positions:
            colors = plt.cm.Set1(np.linspace(0, 1, len(robot_positions)))
            for i, pos in enumerate(robot_positions):
                plt.plot(pos[0], pos[1], 'o', markersize=15, 
                        color=colors[i], markeredgecolor='black', 
                        markeredgewidth=2)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
    def step(self, dt: float):
        """Advance simulation time."""
        self.current_time += dt
