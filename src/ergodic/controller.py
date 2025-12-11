"""
Ergodic control module for generating optimal coverage trajectories.
Based on spectral multi-scale coverage control.
"""

import numpy as np
from typing import Tuple, Callable
from scipy.optimize import minimize


class ErgodicController:
    """
    Ergodic controller that generates trajectories matching a spatial distribution.
    Uses Fourier basis functions to represent distributions.
    """
    
    def __init__(self, map_size: Tuple[float, float], num_basis: int = 10,
                 horizon: int = 50, dt: float = 0.1):
        """
        Initialize ergodic controller.
        
        Args:
            map_size: Map size (width, height) in meters
            num_basis: Number of Fourier basis functions per dimension
            horizon: Planning horizon length
            dt: Time step for planning
        """
        self.map_size = np.array(map_size)
        self.num_basis = num_basis
        self.horizon = horizon
        self.dt = dt
        
        # Fourier basis frequencies
        self.k_indices = self._generate_k_indices()
        
        # Weights for ergodic metric (decay with frequency)
        self.Lambda = self._compute_weights()
        
        # Control limits - INCREASED for better exploration
        self.v_max = 0.8  # m/s (increased from 0.5)
        self.omega_max = 1.5  # rad/s (increased from 1.0)
        
    def _generate_k_indices(self) -> np.ndarray:
        """Generate Fourier basis function indices."""
        indices = []
        for kx in range(self.num_basis):
            for ky in range(self.num_basis):
                indices.append([kx, ky])
        return np.array(indices)
    
    def _compute_weights(self) -> np.ndarray:
        """Compute weights for ergodic metric (decay with frequency)."""
        weights = np.zeros(len(self.k_indices))
        for i, k in enumerate(self.k_indices):
            k_norm = np.linalg.norm(k)
            weights[i] = (1.0 + k_norm ** 2) ** (-1.5)
        return weights
    
    def compute_spatial_distribution(self, target_dist: np.ndarray) -> np.ndarray:
        """
        Compute Fourier coefficients of target distribution.
        
        Args:
            target_dist: 2D probability distribution (must be normalized)
            
        Returns:
            Fourier coefficients
        """
        h, w = target_dist.shape
        coeffs = np.zeros(len(self.k_indices))
        
        for i, k in enumerate(self.k_indices):
            # Create basis function on grid
            x = np.linspace(0, self.map_size[0], w)
            y = np.linspace(0, self.map_size[1], h)
            X, Y = np.meshgrid(x, y, indexing='ij')
            
            basis = self._basis_function(X, Y, k)
            
            # Compute coefficient
            coeffs[i] = np.sum(target_dist * basis) * (self.map_size[0] / w) * (self.map_size[1] / h)
        
        return coeffs
    
    def compute_trajectory_distribution(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Compute Fourier coefficients of trajectory distribution.
        
        Args:
            trajectory: Nx2 array of (x, y) positions
            
        Returns:
            Fourier coefficients
        """
        coeffs = np.zeros(len(self.k_indices))
        N = len(trajectory)
        
        for i, k in enumerate(self.k_indices):
            # Evaluate basis function at trajectory points
            basis_vals = self._basis_function(trajectory[:, 0], trajectory[:, 1], k)
            coeffs[i] = np.mean(basis_vals)
        
        return coeffs
    
    def _basis_function(self, x: np.ndarray, y: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Evaluate Fourier basis function.
        
        Args:
            x, y: Coordinates (can be arrays)
            k: Frequency indices [kx, ky]
            
        Returns:
            Basis function values
        """
        kx, ky = k
        hk = 1.0 / np.sqrt(self.map_size[0] * self.map_size[1])
        
        # Normalization factors
        if kx == 0:
            hk *= 1.0
        else:
            hk *= np.sqrt(2)
        
        if ky == 0:
            hk *= 1.0
        else:
            hk *= np.sqrt(2)
        
        # Basis function
        phi = hk * np.cos(kx * np.pi * x / self.map_size[0]) * \
              np.cos(ky * np.pi * y / self.map_size[1])
        
        return phi
    
    def ergodic_metric(self, trajectory: np.ndarray, target_coeffs: np.ndarray) -> float:
        """
        Compute ergodic metric between trajectory and target distribution.
        
        Args:
            trajectory: Trajectory positions
            target_coeffs: Target distribution Fourier coefficients
            
        Returns:
            Ergodic metric value (lower is better)
        """
        traj_coeffs = self.compute_trajectory_distribution(trajectory)
        diff = traj_coeffs - target_coeffs
        metric = np.sum(self.Lambda * diff ** 2)
        return metric
    
    def plan_trajectory(self, current_state: np.ndarray, target_dist: np.ndarray) -> np.ndarray:
        """
        Plan trajectory to match target distribution.
        
        Args:
            current_state: Current robot state [x, y, theta]
            target_dist: Target spatial distribution (normalized)
            
        Returns:
            Planned control sequence (horizon x 2) [v, omega]
        """
        # Compute target Fourier coefficients
        target_coeffs = self.compute_spatial_distribution(target_dist)
        
        # Better initial guess - move toward highest probability region
        h, w = target_dist.shape
        max_idx = np.unravel_index(np.argmax(target_dist), target_dist.shape)
        target_pos = np.array([
            max_idx[0] * self.map_size[0] / h,
            max_idx[1] * self.map_size[1] / w
        ])
        
        # Direction to target
        to_target = target_pos - current_state[:2]
        angle_to_target = np.arctan2(to_target[1], to_target[0])
        
        # Initial control guess - move toward target
        u0 = np.zeros((self.horizon, 2))
        u0[:, 0] = 0.4  # Forward velocity
        
        # Add some turning toward target
        heading_error = np.arctan2(np.sin(angle_to_target - current_state[2]),
                                   np.cos(angle_to_target - current_state[2]))
        u0[:, 1] = 0.5 * heading_error  # Turn toward target
        
        # Flatten for optimization
        u0_flat = u0.flatten()
        
        # Bounds on controls
        bounds = []
        for _ in range(self.horizon):
            bounds.append((0.0, self.v_max))  # Only forward motion
            bounds.append((-self.omega_max, self.omega_max))  # Angular velocity
        
        # Optimize trajectory with fewer iterations for speed
        def objective(u_flat):
            u = u_flat.reshape((self.horizon, 2))
            trajectory = self._simulate_trajectory(current_state, u)
            
            # Ergodic cost
            erg_cost = self.ergodic_metric(trajectory, target_coeffs)
            
            # Control effort penalty (smooth controls)
            control_cost = 0.001 * np.sum(u ** 2)
            
            # Penalty for low velocity (encourage exploration)
            velocity_penalty = 0.1 * np.sum((self.v_max - u[:, 0]) ** 2)
            
            return erg_cost + control_cost + velocity_penalty
        
        # Optimization - reduced iterations for speed
        result = minimize(objective, u0_flat, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 20, 'ftol': 1e-3})
        
        u_opt = result.x.reshape((self.horizon, 2))
        
        return u_opt
    
    def _simulate_trajectory(self, initial_state: np.ndarray, 
                            controls: np.ndarray) -> np.ndarray:
        """
        Simulate trajectory from initial state with given controls.
        
        Args:
            initial_state: [x, y, theta]
            controls: (N x 2) control sequence [v, omega]
            
        Returns:
            (N x 2) trajectory positions
        """
        state = initial_state.copy()
        trajectory = [state[:2]]
        
        for u in controls:
            v, omega = u
            
            # Differential drive dynamics
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
            
            trajectory.append(state[:2])
        
        return np.array(trajectory)
    
    def get_next_control(self, current_state: np.ndarray, 
                        target_dist: np.ndarray) -> np.ndarray:
        """
        Get next control action (receding horizon).
        
        Args:
            current_state: Current robot state [x, y, theta]
            target_dist: Target distribution
            
        Returns:
            Next control [v, omega]
        """
        control_sequence = self.plan_trajectory(current_state, target_dist)
        return control_sequence[0]


class InformationDistribution:
    """
    Manages the information distribution for ergodic search.
    Combines unexplored regions, features, and detections.
    """
    
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize information distribution.
        
        Args:
            grid_size: Grid size (width, height)
        """
        self.grid_size = grid_size
        
        # Component weights (adaptively adjusted) - better initial balance
        self.w_exploration = 2.0  # Stronger exploration
        self.w_features = 0.3
        self.w_detections = 1.0
        
    def compute(self, unexplored: np.ndarray, 
               feature_density: np.ndarray,
               detection_dist: np.ndarray) -> np.ndarray:
        """
        Compute combined information distribution.
        
        Args:
            unexplored: Unexplored region distribution
            feature_density: Visual feature density
            detection_dist: Detection hypothesis distribution
            
        Returns:
            Combined normalized distribution
        """
        # Ensure all have same shape
        assert unexplored.shape == self.grid_size
        assert feature_density.shape == self.grid_size
        assert detection_dist.shape == self.grid_size
        
        # Combine components
        distribution = (self.w_exploration * unexplored +
                       self.w_features * feature_density +
                       self.w_detections * detection_dist)
        
        # Normalize
        total = distribution.sum()
        if total > 0:
            distribution /= total
        else:
            distribution = np.ones(self.grid_size) / np.prod(self.grid_size)
        
        return distribution
    
    def adapt_weights(self, exploration_percentage: float):
        """
        Adapt component weights based on exploration progress.
        
        Args:
            exploration_percentage: Percentage of map explored (0-100)
        """
        # As exploration progresses, shift from exploration to detection
        progress = exploration_percentage / 100.0
        
        self.w_exploration = max(0.2, 1.0 - progress)
        self.w_features = 0.3 + 0.2 * progress
        self.w_detections = 1.0 + 2.0 * progress
