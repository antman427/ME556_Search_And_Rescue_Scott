"""
Computer vision module for detecting objectives in the environment.
Implements template matching and distance-based confidence scoring.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Represents a detected objective."""
    obj_type: str
    position: np.ndarray
    distance: float
    confidence: float
    timestamp: float


class ObjectDetector:
    """
    Object detector using simulated computer vision.
    Implements coarse (distance-based) and fine (template-based) detection.
    """
    
    def __init__(self, objective_types: List[str]):
        """
        Initialize detector.
        
        Args:
            objective_types: List of objective types to detect
        """
        self.objective_types = objective_types
        
        # Detection parameters - increased ranges
        self.coarse_range = 5.0  # Max range for coarse detection (meters) - was 3.0
        self.fine_range = 2.5    # Max range for fine/confident detection (meters) - was 1.5
        
        # Confidence thresholds
        self.min_confidence = 0.3
        self.confirm_confidence = 0.7  # Lowered from 0.8 to detect more easily
        
        # Detection history for tracking
        self.detection_history: List[Detection] = []
        
    def detect(self, camera_data: dict, robot_pos: np.ndarray,
              current_time: float) -> List[Detection]:
        """
        Detect objectives from camera data.
        
        Args:
            camera_data: Camera observation dict with visible objectives
            robot_pos: Robot position for distance calculation
            current_time: Current simulation time
            
        Returns:
            List of detections with confidence scores
        """
        detections = []
        
        visible_objs = camera_data.get('objectives', [])
        
        for obj_data in visible_objs:
            obj = obj_data['objective']
            distance = obj_data['distance']
            
            # Calculate confidence based on distance
            confidence = self._calculate_confidence(distance, obj.obj_type)
            
            if confidence >= self.min_confidence:
                detection = Detection(
                    obj_type=obj.obj_type,
                    position=obj.position.copy(),
                    distance=distance,
                    confidence=confidence,
                    timestamp=current_time
                )
                detections.append(detection)
                self.detection_history.append(detection)
        
        return detections
    
    def _calculate_confidence(self, distance: float, obj_type: str) -> float:
        """
        Calculate detection confidence based on distance and object type.
        
        Args:
            distance: Distance to object
            obj_type: Type of objective
            
        Returns:
            Confidence score [0, 1]
        """
        # Base confidence decreases with distance
        if distance > self.coarse_range:
            return 0.0
        
        # Linear decay with distance
        base_confidence = 1.0 - (distance / self.coarse_range)
        
        # Add noise to simulate CV uncertainty
        noise = np.random.normal(0, 0.1)
        confidence = np.clip(base_confidence + noise, 0.0, 1.0)
        
        # Fine detection at close range
        if distance < self.fine_range:
            confidence = max(confidence, 0.7)  # Boost confidence when close
        
        return confidence
    
    def is_confirmed(self, detection: Detection) -> bool:
        """
        Check if detection is confident enough for confirmation.
        
        Args:
            detection: Detection to check
            
        Returns:
            True if detection is confirmed
        """
        return detection.confidence >= self.confirm_confidence
    
    def get_recent_detections(self, time_window: float, 
                             current_time: float) -> List[Detection]:
        """
        Get detections within a time window.
        
        Args:
            time_window: Time window in seconds
            current_time: Current time
            
        Returns:
            List of recent detections
        """
        return [d for d in self.detection_history 
                if current_time - d.timestamp < time_window]


class DetectionParticleFilter:
    """
    Particle filter for tracking detected objectives.
    Maintains probability distribution over objective locations.
    """
    
    def __init__(self, map_size: Tuple[float, float], num_particles: int = 500):
        """
        Initialize detection tracker.
        
        Args:
            map_size: Map size (width, height) in meters
            num_particles: Number of particles
        """
        self.map_size = map_size
        self.num_particles = num_particles
        
        # Particles per objective type: {obj_type: [particles, weights]}
        self.particle_sets: Dict[str, Dict] = {}
        
    def initialize_objective(self, obj_type: str):
        """
        Initialize particle set for an objective type.
        
        Args:
            obj_type: Type of objective
        """
        if obj_type in self.particle_sets:
            return
        
        # Initialize particles uniformly
        particles = np.zeros((self.num_particles, 2))
        particles[:, 0] = np.random.uniform(0, self.map_size[0], self.num_particles)
        particles[:, 1] = np.random.uniform(0, self.map_size[1], self.num_particles)
        
        weights = np.ones(self.num_particles) / self.num_particles
        
        self.particle_sets[obj_type] = {
            'particles': particles,
            'weights': weights
        }
    
    def update(self, detection: Detection):
        """
        Update particle filter with a detection.
        
        Args:
            detection: Detection to incorporate
        """
        obj_type = detection.obj_type
        
        if obj_type not in self.particle_sets:
            self.initialize_objective(obj_type)
        
        particles = self.particle_sets[obj_type]['particles']
        weights = self.particle_sets[obj_type]['weights']
        
        # Calculate likelihood of each particle
        distances = np.linalg.norm(particles - detection.position, axis=1)
        
        # Gaussian likelihood based on detection confidence
        sigma = 0.5 / detection.confidence  # Higher confidence = narrower distribution
        likelihoods = np.exp(-0.5 * (distances / sigma) ** 2)
        
        # Update weights
        weights *= likelihoods
        weights += 1e-10  # Avoid zeros
        weights /= weights.sum()
        
        self.particle_sets[obj_type]['weights'] = weights
        
        # Resample if effective sample size is low
        if self._effective_sample_size(weights) < self.num_particles / 2:
            self._resample(obj_type)
    
    def _resample(self, obj_type: str):
        """Resample particles for an objective type."""
        particles = self.particle_sets[obj_type]['particles']
        weights = self.particle_sets[obj_type]['weights']
        
        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        cumsum = np.cumsum(weights)
        
        i, j = 0, 0
        new_particles = np.zeros_like(particles)
        
        while i < self.num_particles:
            if positions[i] < cumsum[j]:
                # Add small noise to resampled particles
                new_particles[i] = particles[j] + np.random.normal(0, 0.1, 2)
                i += 1
            else:
                j += 1
        
        self.particle_sets[obj_type]['particles'] = new_particles
        self.particle_sets[obj_type]['weights'] = np.ones(self.num_particles) / self.num_particles
    
    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """Calculate effective sample size."""
        return 1.0 / np.sum(weights ** 2)
    
    def get_estimate(self, obj_type: str) -> Optional[np.ndarray]:
        """
        Get position estimate for an objective type.
        
        Args:
            obj_type: Type of objective
            
        Returns:
            Estimated position or None if not tracking
        """
        if obj_type not in self.particle_sets:
            return None
        
        particles = self.particle_sets[obj_type]['particles']
        weights = self.particle_sets[obj_type]['weights']
        
        # Weighted mean
        estimate = np.sum(particles * weights[:, np.newaxis], axis=0)
        
        return estimate
    
    def get_distribution(self, obj_type: str, grid_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get probability distribution over a grid.
        
        Args:
            obj_type: Type of objective
            grid_size: Size of output grid (width, height)
            
        Returns:
            2D probability distribution or None
        """
        if obj_type not in self.particle_sets:
            return None
        
        particles = self.particle_sets[obj_type]['particles']
        weights = self.particle_sets[obj_type]['weights']
        
        # Create grid
        distribution = np.zeros(grid_size)
        
        # Bin particles into grid
        x_indices = (particles[:, 0] / self.map_size[0] * grid_size[0]).astype(int)
        y_indices = (particles[:, 1] / self.map_size[1] * grid_size[1]).astype(int)
        
        # Clip to bounds
        x_indices = np.clip(x_indices, 0, grid_size[0] - 1)
        y_indices = np.clip(y_indices, 0, grid_size[1] - 1)
        
        # Accumulate weights in grid
        for i in range(len(particles)):
            distribution[x_indices[i], y_indices[i]] += weights[i]
        
        # Normalize
        if distribution.sum() > 0:
            distribution /= distribution.sum()
        
        return distribution
    
    def get_all_distributions(self, grid_size: Tuple[int, int]) -> np.ndarray:
        """
        Get combined probability distribution for all objectives.
        
        Args:
            grid_size: Size of output grid
            
        Returns:
            Combined 2D probability distribution
        """
        combined = np.zeros(grid_size)
        
        for obj_type in self.particle_sets:
            dist = self.get_distribution(obj_type, grid_size)
            if dist is not None:
                combined += dist
        
        # Normalize
        if combined.sum() > 0:
            combined /= combined.sum()
        
        return combined
