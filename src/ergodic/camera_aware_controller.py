"""
Improved ergodic controller that considers camera field of view.
This ensures the robot not only visits areas but also looks at them properly.
"""

import numpy as np
from typing import Tuple
from src.ergodic.controller import ErgodicController


class CameraAwareErgodicController(ErgodicController):
    """
    Enhanced ergodic controller that plans both position AND heading
    to ensure camera actually observes high-priority regions.
    """
    
    def __init__(self, map_size: Tuple[float, float], num_basis: int = 3,
                 horizon: int = 15, dt: float = 0.1, camera_fov: float = np.pi/2):
        super().__init__(map_size, num_basis, horizon, dt)
        self.camera_fov = camera_fov
        
    def plan_trajectory(self, current_state: np.ndarray, target_dist: np.ndarray) -> np.ndarray:
        """
        Enhanced planning that considers camera direction.
        
        Args:
            current_state: [x, y, theta]
            target_dist: Target distribution
            
        Returns:
            Control sequence [v, omega]
        """
        # Compute target Fourier coefficients
        target_coeffs = self.compute_spatial_distribution(target_dist)
        
        # Find highest probability region
        h, w = target_dist.shape
        max_idx = np.unravel_index(np.argmax(target_dist), target_dist.shape)
        target_pos = np.array([
            max_idx[0] * self.map_size[0] / h,
            max_idx[1] * self.map_size[1] / w
        ])
        
        # Initial guess - move toward AND look at target
        to_target = target_pos - current_state[:2]
        angle_to_target = np.arctan2(to_target[1], to_target[0])
        
        u0 = np.zeros((self.horizon, 2))
        u0[:, 0] = 0.5  # Forward velocity
        
        # Turn toward target more aggressively
        heading_error = np.arctan2(np.sin(angle_to_target - current_state[2]),
                                   np.cos(angle_to_target - current_state[2]))
        u0[:, 1] = 0.8 * heading_error
        
        u0_flat = u0.flatten()
        
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0, self.v_max))
            bounds.append((-self.omega_max, self.omega_max))
        
        def objective(u_flat):
            u = u_flat.reshape((self.horizon, 2))
            trajectory = self._simulate_trajectory(current_state, u)
            
            # Standard ergodic cost
            erg_cost = self.ergodic_metric(trajectory, target_coeffs)
            
            # NEW: Camera coverage cost
            # Penalize if robot doesn't look at high-probability regions
            camera_cost = self._compute_camera_coverage_cost(
                current_state, u, target_dist
            )
            
            # Control effort
            control_cost = 0.001 * np.sum(u ** 2)
            
            # Velocity penalty
            velocity_penalty = 0.1 * np.sum((self.v_max - u[:, 0]) ** 2)
            
            return erg_cost + 0.5 * camera_cost + control_cost + velocity_penalty
        
        from scipy.optimize import minimize
        result = minimize(objective, u0_flat, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 20, 'ftol': 1e-3})
        
        u_opt = result.x.reshape((self.horizon, 2))
        return u_opt
    
    def _compute_camera_coverage_cost(self, initial_state: np.ndarray,
                                     controls: np.ndarray,
                                     target_dist: np.ndarray) -> float:
        """
        Compute cost based on how well camera observes target distribution.
        
        Lower cost = camera points at high-priority regions more often.
        """
        h, w = target_dist.shape
        state = initial_state.copy()
        total_cost = 0.0
        
        for u in controls:
            v, omega = u
            
            # Update state
            if abs(omega) < 1e-6:
                dx = v * np.cos(state[2]) * self.dt
                dy = v * np.sin(state[2]) * self.dt
                dtheta = 0
            else:
                R = v / omega
                dx = R * (np.sin(state[2] + omega * self.dt) - np.sin(state[2]))
                dy = -R * (np.cos(state[2] + omega * self.dt) - np.cos(state[2]))
                dtheta = omega * self.dt
            
            state = state + np.array([dx, dy, dtheta])
            state[2] = np.arctan2(np.sin(state[2]), np.cos(state[2]))
            
            # Compute what camera sees at this state
            camera_coverage = 0.0
            
            # Sample points in camera FOV
            for angle_offset in np.linspace(-self.camera_fov/2, self.camera_fov/2, 10):
                cam_angle = state[2] + angle_offset
                
                # Point in camera view (at 3m range)
                view_x = state[0] + 3.0 * np.cos(cam_angle)
                view_y = state[1] + 3.0 * np.sin(cam_angle)
                
                # Convert to grid
                grid_x = int(view_x / self.map_size[0] * h)
                grid_y = int(view_y / self.map_size[1] * w)
                
                # Check if in bounds and get probability
                if 0 <= grid_x < h and 0 <= grid_y < w:
                    camera_coverage += target_dist[grid_x, grid_y]
            
            # Cost is inverse of coverage (we want high coverage)
            # If camera sees high-prob regions, cost is low
            total_cost += 1.0 / (camera_coverage + 0.01)
        
        return total_cost / len(controls)
