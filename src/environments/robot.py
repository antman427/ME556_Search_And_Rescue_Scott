"""
Robot module for simulating differential drive robots with sensors.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RobotConfig:
    """Configuration parameters for a robot."""
    max_linear_velocity: float = 0.8  # m/s (increased)
    max_angular_velocity: float = 1.5  # rad/s (increased)
    wheel_base: float = 0.3  # meters
    sensor_range: float = 5.0  # meters
    camera_fov: float = np.pi / 2  # radians (wider FOV - was pi/3)
    camera_range: float = 5.0  # meters (increased from 3.0)
    lidar_rays: int = 360
    process_noise_std: float = 0.01
    

class Robot:
    """
    Differential drive robot with LIDAR and camera sensors.
    """
    
    def __init__(self, robot_id: int, initial_position: np.ndarray, 
                 initial_heading: float = 0.0, config: Optional[RobotConfig] = None):
        """
        Initialize robot.
        
        Args:
            robot_id: Unique robot identifier
            initial_position: Starting (x, y) position in meters
            initial_heading: Starting heading in radians
            config: Robot configuration parameters
        """
        self.id = robot_id
        self.config = config if config else RobotConfig()
        
        # State: [x, y, theta]
        self.state = np.array([initial_position[0], 
                               initial_position[1], 
                               initial_heading])
        
        # Control inputs: [v, omega]
        self.control = np.array([0.0, 0.0])
        
        # Trajectory history
        self.trajectory = [self.state[:2].copy()]
        
        # Sensor data
        self.lidar_data = None
        self.camera_data = None
        
    @property
    def position(self) -> np.ndarray:
        """Get current position (x, y)."""
        return self.state[:2]
    
    @property
    def heading(self) -> float:
        """Get current heading in radians."""
        return self.state[2]
    
    def set_control(self, linear_velocity: float, angular_velocity: float):
        """
        Set control inputs with velocity limits.
        
        Args:
            linear_velocity: Desired linear velocity (m/s)
            angular_velocity: Desired angular velocity (rad/s)
        """
        self.control[0] = np.clip(linear_velocity, 
                                  -self.config.max_linear_velocity,
                                  self.config.max_linear_velocity)
        self.control[1] = np.clip(angular_velocity,
                                   -self.config.max_angular_velocity,
                                   self.config.max_angular_velocity)
    
    def step(self, dt: float, environment):
        """
        Update robot state using differential drive dynamics.
        
        Args:
            dt: Time step in seconds
            environment: Environment object for collision checking
        """
        # Current state
        x, y, theta = self.state
        v, omega = self.control
        
        # Check for obstacles ahead using simple ray cast
        look_ahead_dist = 0.3  # meters
        look_ahead_pos = np.array([
            x + look_ahead_dist * np.cos(theta),
            y + look_ahead_dist * np.sin(theta)
        ])
        
        # If obstacle ahead, turn away
        if not environment.is_free(look_ahead_pos):
            # Turn sharply away from obstacle
            v = 0.2  # Slow down
            omega = self.config.max_angular_velocity * 0.7  # Turn
        
        # Add process noise to controls (input noise, not output noise)
        # Noise proportional to control magnitude for realism
        v_noise = np.random.normal(0, self.config.process_noise_std * abs(v))
        omega_noise = np.random.normal(0, self.config.process_noise_std * abs(omega))
        v_noisy = v + v_noise
        omega_noisy = omega + omega_noise
        
        # Differential drive kinematics with noisy controls
        if abs(omega_noisy) < 1e-6:
            # Straight line motion
            dx = v_noisy * np.cos(theta) * dt
            dy = v_noisy * np.sin(theta) * dt
            dtheta = 0
        else:
            # Arc motion
            R = v_noisy / omega_noisy  # Radius of curvature
            dx = R * (np.sin(theta + omega_noisy * dt) - np.sin(theta))
            dy = -R * (np.cos(theta + omega_noisy * dt) - np.cos(theta))
            dtheta = omega_noisy * dt
        
        # Update state
        new_state = self.state + np.array([dx, dy, dtheta])
        new_state[2] = np.arctan2(np.sin(new_state[2]), np.cos(new_state[2]))  # Wrap angle
        
        # Check if new position is valid (not in collision)
        if environment.is_free(new_state[:2]):
            self.state = new_state
            self.trajectory.append(self.state[:2].copy())
        else:
            # If collision, try to back up and turn
            self.control = np.array([-0.2, self.config.max_angular_velocity * 0.5])
            # Don't update position if in collision
    
    def sense(self, environment):
        """
        Update sensor readings from environment.
        
        Args:
            environment: Environment object
        """
        # LIDAR scan
        self.lidar_data = environment.get_lidar_scan(
            self.position,
            num_rays=self.config.lidar_rays,
            max_range=self.config.sensor_range,
            noise_std=0.05
        )
        
        # Camera view
        self.camera_data = environment.get_camera_view(
            self.position,
            self.heading,
            fov=self.config.camera_fov,
            range_limit=self.config.camera_range
        )
    
    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory as numpy array."""
        return np.array(self.trajectory)
    
    def move_to_goal(self, goal: np.ndarray, dt: float) -> bool:
        """
        Simple proportional controller to move toward a goal.
        
        Args:
            goal: Target (x, y) position
            dt: Time step
            
        Returns:
            True if goal reached (within 0.1m)
        """
        # Vector to goal
        to_goal = goal - self.position
        distance = np.linalg.norm(to_goal)
        
        # Check if reached
        if distance < 0.1:
            self.set_control(0.0, 0.0)
            return True
        
        # Desired heading
        desired_heading = np.arctan2(to_goal[1], to_goal[0])
        
        # Heading error
        heading_error = np.arctan2(np.sin(desired_heading - self.heading),
                                   np.cos(desired_heading - self.heading))
        
        # Proportional control - more aggressive
        k_linear = 1.0  # Linear velocity gain (increased)
        k_angular = 3.0  # Angular velocity gain (increased)
        
        # If heading error is large, rotate in place
        if abs(heading_error) > np.pi / 4:  # Changed from pi/6 to pi/4
            v = 0.2 * distance  # Some forward motion even while turning
            omega = k_angular * heading_error
        else:
            v = k_linear * distance
            omega = k_angular * heading_error
        
        self.set_control(v, omega)
        return False
    
    def __repr__(self):
        return (f"Robot(id={self.id}, pos=({self.state[0]:.2f}, {self.state[1]:.2f}), "
                f"heading={np.degrees(self.state[2]):.1f}°)")


class DifferentialDriveModel:
    """
    Differential drive kinematic model for prediction and planning.
    """
    
    @staticmethod
    def predict(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Predict next state given current state and control.
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
            dt: Time step
            
        Returns:
            Predicted next state
        """
        x, y, theta = state
        v, omega = control
        
        if abs(omega) < 1e-6:
            dx = v * np.cos(theta) * dt
            dy = v * np.sin(theta) * dt
            dtheta = 0
        else:
            R = v / omega
            dx = R * (np.sin(theta + omega * dt) - np.sin(theta))
            dy = -R * (np.cos(theta + omega * dt) - np.cos(theta))
            dtheta = omega * dt
        
        next_state = state + np.array([dx, dy, dtheta])
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        
        return next_state
    
    @staticmethod
    def jacobian(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute Jacobian of motion model with respect to state.
        
        Args:
            state: [x, y, theta]
            control: [v, omega]
            dt: Time step
            
        Returns:
            3x3 Jacobian matrix
        """
        theta = state[2]
        v, omega = control
        
        J = np.eye(3)
        
        if abs(omega) < 1e-6:
            J[0, 2] = -v * np.sin(theta) * dt
            J[1, 2] = v * np.cos(theta) * dt
        else:
            J[0, 2] = (v / omega) * (np.cos(theta + omega * dt) - np.cos(theta))
            J[1, 2] = (v / omega) * (np.sin(theta + omega * dt) - np.sin(theta))
        
        return J
