"""
Quick integration test to verify all components work together.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.environment import Environment
from src.environments.robot import Robot, RobotConfig
from src.slam.slam import VisualSLAM
from src.vision.detector import ObjectDetector, DetectionParticleFilter
from src.ergodic.controller import ErgodicController, InformationDistribution


def test_environment():
    """Test environment creation and basic functions."""
    print("Testing Environment...", end=" ")
    env = Environment(map_size=(10.0, 10.0), resolution=0.1)
    env.add_objective(np.array([5.0, 5.0]), 'person')
    assert len(env.objectives) == 1
    assert env.is_free(np.array([1.0, 1.0]))
    print("PASS")


def test_robot():
    """Test robot creation and motion."""
    print("Testing Robot...", end=" ")
    robot = Robot(robot_id=0, initial_position=np.array([1.0, 1.0]))
    robot.set_control(0.5, 0.0)
    env = Environment(map_size=(10.0, 10.0))
    robot.step(0.1, env)
    assert robot.position[0] > 1.0  # Should have moved forward
    print("PASS")


def test_slam():
    """Test SLAM initialization."""
    print("Testing SLAM...", end=" ")
    slam = VisualSLAM(map_size=(10.0, 10.0), resolution=0.1, num_particles=100)
    pose = slam.get_pose()
    assert pose.shape == (3,)
    print("PASS")


def test_detector():
    """Test object detector."""
    print("Testing Detector...", end=" ")
    detector = ObjectDetector(objective_types=['person', 'pet'])
    camera_data = {'objectives': []}
    detections = detector.detect(camera_data, np.array([1.0, 1.0]), 0.0)
    assert len(detections) == 0
    print("PASS")


def test_detection_filter():
    """Test detection particle filter."""
    print("Testing Detection Filter...", end=" ")
    det_filter = DetectionParticleFilter(map_size=(10.0, 10.0), num_particles=100)
    det_filter.initialize_objective('person')
    estimate = det_filter.get_estimate('person')
    assert estimate is not None
    assert estimate.shape == (2,)
    print("PASS")


def test_ergodic_controller():
    """Test ergodic controller."""
    print("Testing Ergodic Controller...", end=" ")
    controller = ErgodicController(map_size=(10.0, 10.0), num_basis=3, 
                                  horizon=10, dt=0.1)
    
    # Create simple target distribution
    target_dist = np.zeros((100, 100))
    target_dist[40:60, 40:60] = 1.0
    target_dist /= target_dist.sum()
    
    # Test trajectory planning
    current_state = np.array([1.0, 1.0, 0.0])
    control = controller.get_next_control(current_state, target_dist)
    assert control.shape == (2,)
    print("PASS")


def test_information_distribution():
    """Test information distribution."""
    print("Testing Information Distribution...", end=" ")
    info_dist = InformationDistribution(grid_size=(100, 100))
    
    unexplored = np.random.rand(100, 100)
    features = np.random.rand(100, 100)
    detections = np.random.rand(100, 100)
    
    distribution = info_dist.compute(unexplored, features, detections)
    assert distribution.shape == (100, 100)
    assert np.abs(distribution.sum() - 1.0) < 1e-6  # Should be normalized
    print("PASS")


def test_integration():
    """Test basic integration of components."""
    print("Testing Integration...", end=" ")
    
    # Create environment
    env = Environment(map_size=(10.0, 10.0), resolution=0.1)
    env.add_objective(np.array([5.0, 5.0]), 'person')
    
    # Create robot
    robot = Robot(robot_id=0, initial_position=np.array([1.0, 1.0]))
    
    # Create SLAM
    slam = VisualSLAM(map_size=(10.0, 10.0), resolution=0.1, num_particles=50)
    
    # Create detector
    detector = ObjectDetector(['person'])
    
    # Run a few steps
    for _ in range(5):
        robot.sense(env)
        lidar_angles = np.linspace(0, 2*np.pi, robot.config.lidar_rays, endpoint=False)
        slam.update(robot.control, robot.lidar_data, lidar_angles, 0.1)
        detections = detector.detect(robot.camera_data, robot.position, env.current_time)
        robot.set_control(0.2, 0.1)
        robot.step(0.1, env)
        env.step(0.1)
    
    print("PASS")


def main():
    print("\n" + "="*60)
    print("Running Component Tests")
    print("="*60 + "\n")
    
    try:
        test_environment()
        test_robot()
        test_slam()
        test_detector()
        test_detection_filter()
        test_ergodic_controller()
        test_information_distribution()
        test_integration()
        
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60 + "\n")
        return 0
        
    except Exception as e:
        print(f"\nFAIL: Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
