"""
Multi-Robot Coordinator

Manages multiple robots searching cooperatively with:
1. Shared map (merged from all robots)
2. Distributed search regions (avoid redundant coverage)
3. Shared detections (all robots know what's been found)
4. Collision avoidance (robots don't bump into each other)
"""

import numpy as np
from typing import List, Tuple, Dict
from src.environments.environment import Environment
from src.environments.robot import Robot
from src.slam.slam import VisualSLAM, OccupancyGridMap
from src.vision.detector import ObjectDetector, DetectionParticleFilter
from src.ergodic.camera_aware_controller import CameraAwareErgodicController
from src.ergodic.controller import InformationDistribution
from scipy.ndimage import zoom


class MultiRobotCoordinator:
    """
    Coordinates multiple robots for cooperative search.
    """
    
    def __init__(self, num_robots: int, environment: Environment,
                 spawn_positions: List[np.ndarray], map_size: Tuple[float, float],
                 resolution: float = 0.1, dt: float = 0.1):
        """
        Initialize multi-robot system.
        
        Args:
            num_robots: Number of robots
            environment: Shared environment
            spawn_positions: Initial positions for each robot
            map_size: (width, height) in meters
            resolution: Map resolution
            dt: Time step
        """
        self.num_robots = num_robots
        self.environment = environment
        self.map_size = map_size
        self.resolution = resolution
        self.dt = dt
        
        # Create robots
        self.robots = []
        for i, pos in enumerate(spawn_positions):
            robot = Robot(robot_id=i, initial_position=pos, initial_heading=0.0)
            self.robots.append(robot)
        
        # Create SLAM system for each robot
        self.slam_systems = []
        for i in range(num_robots):
            slam = VisualSLAM(map_size=map_size, resolution=resolution, num_particles=200)
            self.slam_systems.append(slam)
        
        # Global merged map
        self.global_map = OccupancyGridMap(map_size=map_size, resolution=resolution)
        
        # Shared vision system
        self.detector = ObjectDetector(objective_types=['person', 'pet'])
        self.detection_filter = DetectionParticleFilter(map_size=map_size, num_particles=500)
        for obj_type in ['person', 'pet']:
            self.detection_filter.initialize_objective(obj_type)
        
        # Shared detections list
        self.global_detections = []
        
        # Controllers for each robot
        self.controllers = []
        self.info_dists = []
        for i in range(num_robots):
            controller = CameraAwareErgodicController(
                map_size=map_size,
                num_basis=3,
                horizon=15,
                dt=dt,
                camera_fov=self.robots[i].config.camera_fov
            )
            self.controllers.append(controller)
            
            info_dist = InformationDistribution(grid_size=(50, 50))
            self.info_dists.append(info_dist)
        
        # Target distributions for each robot (modified for coordination)
        self.target_distributions = [None] * num_robots
        
        # Time tracking
        self.current_time = 0.0
        
        # Metrics
        self.metrics = {
            'time': [],
            'exploration': [],
            'detections': [],
            'robot_positions': [[] for _ in range(num_robots)]
        }
    
    def merge_maps(self):
        """
        Merge occupancy maps from all robots into global map.
        Uses average of log-odds for consensus mapping.
        """
        # Start with zeros
        merged_log_odds = np.zeros_like(self.global_map.log_odds)
        merged_explored = np.zeros_like(self.global_map.explored, dtype=bool)
        
        # Accumulate from all robots
        for slam in self.slam_systems:
            merged_log_odds += slam.map.log_odds
            merged_explored = np.logical_or(merged_explored, slam.map.explored)
        
        # Average log-odds (only where explored)
        count_map = np.sum([slam.map.explored for slam in self.slam_systems], axis=0)
        count_map[count_map == 0] = 1  # Avoid division by zero
        
        merged_log_odds = merged_log_odds / count_map
        
        # Update global map
        self.global_map.log_odds = merged_log_odds
        self.global_map.explored = merged_explored
        self.global_map._update_occupancy()
    
    def distribute_search_regions(self):
        """
        Assign different search regions to each robot to avoid redundant coverage.
        
        Strategy: Reduce target probability near other robots' positions and
        recent trajectories.
        """
        # Get base information distribution (same for all)
        unexplored = self.global_map.get_unexplored_regions()
        zoom_factor = (50 / unexplored.shape[0], 50 / unexplored.shape[1])
        unexplored_resized = zoom(unexplored, zoom_factor, order=1)
        
        feature_density = np.abs(np.gradient(self.global_map.occupancy)[0]) + \
                         np.abs(np.gradient(self.global_map.occupancy)[1])
        feature_density = zoom(feature_density, zoom_factor, order=1)
        feature_density /= (feature_density.sum() + 1e-10)
        
        detection_dist = self.detection_filter.get_all_distributions(grid_size=(50, 50))
        
        exploration_pct = self.global_map.get_exploration_percentage()
        
        # Create grid for calculating distances
        h, w = 50, 50
        X, Y = np.meshgrid(
            np.linspace(0, self.map_size[0], w),
            np.linspace(0, self.map_size[1], h)
        )
        
        # For each robot, create modified distribution
        for i in range(self.num_robots):
            # Start with base distribution
            self.info_dists[i].adapt_weights(exploration_pct)
            base_dist = self.info_dists[i].compute(
                unexplored_resized, 
                feature_density, 
                detection_dist
            )
            
            # Reduce probability near other robots
            modified_dist = base_dist.copy()
            
            for j in range(self.num_robots):
                if i != j:
                    other_robot = self.robots[j]
                    other_pos = other_robot.position
                    
                    # Gaussian around other robot's current position
                    dist_to_other = np.sqrt(
                        (X - other_pos[0])**2 + 
                        (Y - other_pos[1])**2
                    )
                    
                    # Coverage radius: 3 meters
                    coverage = np.exp(-dist_to_other**2 / (2 * 3.0**2))
                    
                    # Reduce probability in covered region
                    # Keep 30% to allow overlap if needed
                    modified_dist = modified_dist * (1 - 0.7 * coverage)
                    
                    # Also reduce near other robot's recent trajectory
                    if len(other_robot.trajectory) > 10:
                        recent_traj = other_robot.trajectory[-50:]
                        for pos in recent_traj:
                            dist_to_traj = np.sqrt(
                                (X - pos[0])**2 + 
                                (Y - pos[1])**2
                            )
                            traj_coverage = np.exp(-dist_to_traj**2 / (2 * 2.0**2))
                            modified_dist = modified_dist * (1 - 0.5 * traj_coverage)
            
            # Normalize
            modified_dist = modified_dist / (modified_dist.sum() + 1e-10)
            
            # Store for this robot
            self.target_distributions[i] = modified_dist
    
    def check_collisions(self, robot_idx: int, new_position: np.ndarray) -> bool:
        """
        Check if robot would collide with other robots.
        
        Args:
            robot_idx: Index of robot to check
            new_position: Proposed new position
            
        Returns:
            True if safe, False if collision
        """
        min_separation = 0.4  # meters (reduced from 0.5 for better mobility)
        
        for j, other_robot in enumerate(self.robots):
            if j != robot_idx:
                distance = np.linalg.norm(new_position - other_robot.position)
                if distance < min_separation:
                    return False
        
        return True
    
    def step(self):
        """
        Execute one time step for all robots.
        """
        # 1. Each robot senses
        for robot in self.robots:
            robot.sense(self.environment)
        
        # 2. Update each robot's local SLAM
        lidar_angles = np.linspace(0, 2*np.pi, self.robots[0].config.lidar_rays, 
                                   endpoint=False)
        
        for i, (robot, slam) in enumerate(zip(self.robots, self.slam_systems)):
            slam.update(robot.control, robot.lidar_data, lidar_angles, self.dt)
        
        # 3. Merge maps
        self.merge_maps()
        
        # 4. Process detections (shared across all robots)
        for i, robot in enumerate(self.robots):
            detections = self.detector.detect(
                robot.camera_data, 
                robot.position, 
                self.current_time
            )
            
            # Add to global detections
            self.global_detections.extend(detections)
            
            # Update shared detection filter
            for detection in detections:
                self.detection_filter.update(detection)
                
                # Check if any objective discovered
                for obj in self.environment.objectives:
                    if (not obj.discovered and 
                        np.linalg.norm(obj.position - detection.position) < 0.5 and
                        self.detector.is_confirmed(detection)):
                        obj.discovered = True
                        obj.discovered_time = self.current_time
                        obj.discovered_by = i  # Track which robot found it
        
        # 5. Distribute search regions
        self.distribute_search_regions()
        
        # 6. Plan and execute controls for each robot
        for i, robot in enumerate(self.robots):
            robot_state = np.array([
                robot.position[0], 
                robot.position[1], 
                robot.heading
            ])
            
            # Get control from camera-aware controller
            control = self.controllers[i].get_next_control(
                robot_state,
                self.target_distributions[i]
            )
            
            robot.set_control(control[0], control[1])
            
            # Check collision before moving
            # Predict new position
            v, omega = control
            if abs(omega) < 1e-6:
                dx = v * np.cos(robot.heading) * self.dt
                dy = v * np.sin(robot.heading) * self.dt
            else:
                R = v / omega
                dx = R * (np.sin(robot.heading + omega * self.dt) - np.sin(robot.heading))
                dy = -R * (np.cos(robot.heading + omega * self.dt) - np.cos(robot.heading))
            
            predicted_pos = robot.position + np.array([dx, dy])
            
            # Only move if no collision
            if self.check_collisions(i, predicted_pos):
                robot.step(self.dt, self.environment)
            else:
                # Collision detected - turn away
                robot.set_control(0.2, 1.0)  # Slow forward + turn
                robot.step(self.dt, self.environment)
        
        # 7. Update time
        self.current_time += self.dt
        self.environment.step(self.dt)
        
        # 8. Record metrics
        self.metrics['time'].append(self.current_time)
        self.metrics['exploration'].append(self.global_map.get_exploration_percentage())
        self.metrics['detections'].append(
            sum([obj.discovered for obj in self.environment.objectives])
        )
        
        for i, robot in enumerate(self.robots):
            self.metrics['robot_positions'][i].append(robot.position.copy())
    
    def all_objectives_found(self) -> bool:
        """Check if all objectives have been discovered."""
        return all([obj.discovered for obj in self.environment.objectives])
    
    def get_results(self) -> Dict:
        """
        Get comprehensive results.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            'time': self.current_time,
            'exploration': self.global_map.get_exploration_percentage(),
            'objectives_found': sum([obj.discovered for obj in self.environment.objectives]),
            'total_objectives': len(self.environment.objectives),
            'robots': self.robots,
            'global_map': self.global_map,
            'metrics': self.metrics,
            'objectives': self.environment.objectives
        }
