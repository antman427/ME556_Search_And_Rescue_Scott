"""
SLAM (Simultaneous Localization and Mapping) module.
Implements occupancy grid mapping with particle filter localization.
"""

import numpy as np
from typing import Tuple, List
from scipy.stats import norm


class OccupancyGridMap:
    """
    2D occupancy grid map using log-odds representation.
    """
    
    def __init__(self, map_size: Tuple[float, float], resolution: float = 0.1):
        """
        Initialize occupancy grid.
        
        Args:
            map_size: Physical size (width, height) in meters
            resolution: Grid resolution in meters per cell
        """
        self.map_size = map_size
        self.resolution = resolution
        self.grid_size = (int(map_size[0] / resolution),
                         int(map_size[1] / resolution))
        
        # Log-odds representation (0 = unknown, positive = occupied, negative = free)
        self.log_odds = np.zeros(self.grid_size, dtype=np.float32)
        
        # Occupancy probability (0 to 1)
        self.occupancy = 0.5 * np.ones(self.grid_size, dtype=np.float32)
        
        # Exploration state (True if cell has been observed)
        self.explored = np.zeros(self.grid_size, dtype=bool)
        
        # Log-odds update parameters
        self.log_odds_occ = np.log(0.7 / 0.3)  # Occupied update
        self.log_odds_free = np.log(0.3 / 0.7)  # Free update
        self.log_odds_max = 10.0
        self.log_odds_min = -10.0
        
    def world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(position[0] / self.resolution)
        grid_y = int(position[1] / self.resolution)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid coordinates to world coordinates."""
        x = (grid_pos[0] + 0.5) * self.resolution
        y = (grid_pos[1] + 0.5) * self.resolution
        return np.array([x, y])
    
    def in_bounds(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is within bounds."""
        return (0 <= grid_pos[0] < self.grid_size[0] and
                0 <= grid_pos[1] < self.grid_size[1])
    
    def update_from_lidar(self, robot_pos: np.ndarray, robot_heading: float,
                         ranges: np.ndarray, angles: np.ndarray):
        """
        Update map from LIDAR scan using ray tracing.
        
        Args:
            robot_pos: Robot position (x, y)
            robot_heading: Robot heading in radians
            ranges: Array of range measurements
            angles: Array of angles relative to robot heading
        """
        robot_grid = self.world_to_grid(robot_pos)
        
        for range_val, angle in zip(ranges, angles):
            if np.isnan(range_val) or range_val <= 0:
                continue
            
            # Absolute angle
            abs_angle = robot_heading + angle
            
            # End point of ray
            end_x = robot_pos[0] + range_val * np.cos(abs_angle)
            end_y = robot_pos[1] + range_val * np.sin(abs_angle)
            end_grid = self.world_to_grid(np.array([end_x, end_y]))
            
            # Get cells along ray using Bresenham's algorithm
            cells = self._bresenham_line(robot_grid, end_grid)
            
            # Update cells along ray as free
            for i, cell in enumerate(cells[:-1]):
                if self.in_bounds(cell):
                    self.log_odds[cell] += self.log_odds_free
                    self.log_odds[cell] = np.clip(self.log_odds[cell],
                                                   self.log_odds_min,
                                                   self.log_odds_max)
                    self.explored[cell] = True
            
            # Update endpoint as occupied
            if self.in_bounds(end_grid):
                self.log_odds[end_grid] += self.log_odds_occ
                self.log_odds[end_grid] = np.clip(self.log_odds[end_grid],
                                                   self.log_odds_min,
                                                   self.log_odds_max)
                self.explored[end_grid] = True
        
        # Update occupancy probabilities
        self._update_occupancy()
    
    def _bresenham_line(self, start: Tuple[int, int], 
                       end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Bresenham's line algorithm for ray tracing.
        
        Args:
            start: Starting grid cell (x, y)
            end: Ending grid cell (x, y)
            
        Returns:
            List of grid cells along the line
        """
        x0, y0 = start
        x1, y1 = end
        
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            cells.append((x, y))
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return cells
    
    def _update_occupancy(self):
        """Update occupancy probabilities from log-odds."""
        self.occupancy = 1.0 / (1.0 + np.exp(-self.log_odds))
    
    def get_unexplored_regions(self) -> np.ndarray:
        """
        Get probability distribution over unexplored regions.
        
        Returns:
            Normalized probability distribution
        """
        unexplored = (~self.explored).astype(np.float32)
        
        # Normalize
        total = unexplored.sum()
        if total > 0:
            unexplored /= total
        
        return unexplored
    
    def get_exploration_percentage(self) -> float:
        """Get percentage of map explored."""
        return 100.0 * self.explored.sum() / self.explored.size
    
    def is_occupied(self, position: np.ndarray, threshold: float = 0.6) -> bool:
        """
        Check if a position is occupied.
        
        Args:
            position: World position (x, y)
            threshold: Occupancy threshold
            
        Returns:
            True if occupied
        """
        grid_pos = self.world_to_grid(position)
        if not self.in_bounds(grid_pos):
            return True  # Out of bounds considered occupied
        
        return self.occupancy[grid_pos] > threshold


class ParticleFilterLocalizer:
    """
    Particle filter for robot localization within the map.
    """
    
    def __init__(self, num_particles: int = 1000, 
                 map_size: Tuple[float, float] = (10.0, 10.0)):
        """
        Initialize particle filter.
        
        Args:
            num_particles: Number of particles
            map_size: Map size for initialization
        """
        self.num_particles = num_particles
        self.map_size = map_size
        
        # Particles: [x, y, theta, weight]
        self.particles = np.zeros((num_particles, 4))
        self._initialize_particles()
        
        # Motion noise
        self.motion_noise_std = np.array([0.05, 0.05, 0.02])  # [x, y, theta]
        
        # Measurement noise
        self.measurement_noise_std = 0.1
        
    def _initialize_particles(self):
        """Initialize particles uniformly in the map."""
        self.particles[:, 0] = np.random.uniform(0, self.map_size[0], self.num_particles)
        self.particles[:, 1] = np.random.uniform(0, self.map_size[1], self.num_particles)
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.particles[:, 3] = 1.0 / self.num_particles  # Equal weights
    
    def predict(self, control: np.ndarray, dt: float):
        """
        Prediction step: propagate particles using motion model.
        
        Args:
            control: Control input [v, omega]
            dt: Time step
        """
        v, omega = control
        
        for i in range(self.num_particles):
            x, y, theta = self.particles[i, :3]
            
            # Add motion noise
            v_noisy = v + np.random.normal(0, self.motion_noise_std[0])
            omega_noisy = omega + np.random.normal(0, self.motion_noise_std[2])
            
            # Update particle
            if abs(omega_noisy) < 1e-6:
                dx = v_noisy * np.cos(theta) * dt
                dy = v_noisy * np.sin(theta) * dt
                dtheta = 0
            else:
                R = v_noisy / omega_noisy
                dx = R * (np.sin(theta + omega_noisy * dt) - np.sin(theta))
                dy = -R * (np.cos(theta + omega_noisy * dt) - np.cos(theta))
                dtheta = omega_noisy * dt
            
            self.particles[i, 0] += dx
            self.particles[i, 1] += dy
            self.particles[i, 2] += dtheta
            
            # Wrap angle
            self.particles[i, 2] = np.arctan2(np.sin(self.particles[i, 2]),
                                              np.cos(self.particles[i, 2]))
    
    def update(self, measurement: np.ndarray, occupancy_map: OccupancyGridMap):
        """
        Update step: reweight particles based on LIDAR measurement.
        
        Args:
            measurement: LIDAR ranges
            occupancy_map: Current occupancy map
        """
        # Simplified: weight based on correlation with map
        # In full implementation, would ray-trace expected measurements
        
        for i in range(self.num_particles):
            pos = self.particles[i, :2]
            
            # Check if position is in free space
            if occupancy_map.is_occupied(pos, threshold=0.7):
                self.particles[i, 3] *= 0.1  # Low weight if in occupied space
            else:
                self.particles[i, 3] *= 1.0  # Keep weight
        
        # Normalize weights
        weight_sum = self.particles[:, 3].sum()
        if weight_sum > 0:
            self.particles[:, 3] /= weight_sum
        else:
            # Reset if all weights zero
            self.particles[:, 3] = 1.0 / self.num_particles
    
    def resample(self):
        """Resample particles using systematic resampling."""
        weights = self.particles[:, 3]
        
        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumulative_sum = np.cumsum(weights)
        
        i, j = 0, 0
        new_particles = np.zeros_like(self.particles)
        
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        
        self.particles = new_particles
        self.particles[:, 3] = 1.0 / self.num_particles  # Reset weights
    
    def get_estimate(self) -> np.ndarray:
        """
        Get state estimate (weighted mean of particles).
        
        Returns:
            Estimated state [x, y, theta]
        """
        weights = self.particles[:, 3]
        
        x = np.sum(weights * self.particles[:, 0])
        y = np.sum(weights * self.particles[:, 1])
        
        # Circular mean for angle
        theta = np.arctan2(
            np.sum(weights * np.sin(self.particles[:, 2])),
            np.sum(weights * np.cos(self.particles[:, 2]))
        )
        
        return np.array([x, y, theta])


class VisualSLAM:
    """
    Visual SLAM system combining occupancy mapping and particle filter localization.
    """
    
    def __init__(self, map_size: Tuple[float, float] = (10.0, 10.0),
                 resolution: float = 0.1, num_particles: int = 500):
        """
        Initialize SLAM system.
        
        Args:
            map_size: Map size in meters
            resolution: Grid resolution
            num_particles: Number of particles for localization
        """
        self.map = OccupancyGridMap(map_size, resolution)
        self.localizer = ParticleFilterLocalizer(num_particles, map_size)
        
    def update(self, control: np.ndarray, lidar_ranges: np.ndarray,
              lidar_angles: np.ndarray, dt: float):
        """
        Full SLAM update: predict, sense, update map.
        
        Args:
            control: Control input [v, omega]
            lidar_ranges: LIDAR range measurements
            lidar_angles: LIDAR angles
            dt: Time step
        """
        # Prediction
        self.localizer.predict(control, dt)
        
        # Get current pose estimate
        pose = self.localizer.get_estimate()
        
        # Update map
        self.map.update_from_lidar(pose[:2], pose[2], lidar_ranges, lidar_angles)
        
        # Update localization
        self.localizer.update(lidar_ranges, self.map)
        
        # Resample if needed
        if self._effective_sample_size() < self.localizer.num_particles / 2:
            self.localizer.resample()
    
    def _effective_sample_size(self) -> float:
        """Calculate effective sample size."""
        weights = self.localizer.particles[:, 3]
        return 1.0 / np.sum(weights ** 2)
    
    def get_pose(self) -> np.ndarray:
        """Get current pose estimate."""
        return self.localizer.get_estimate()
    
    def get_map(self) -> OccupancyGridMap:
        """Get current map."""
        return self.map
