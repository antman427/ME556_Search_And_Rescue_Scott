"""
Fast demo using greedy controller (baseline).
This should run much faster and show the robot actually moving and finding objectives.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.environment import Environment
from src.environments.robot import Robot
from src.slam.slam import VisualSLAM
from src.vision.detector import ObjectDetector, DetectionParticleFilter
from src.utils.baselines import GreedyController
from src.ergodic.controller import InformationDistribution


def create_simple_map(size=(10, 10), resolution=0.1):
    """Create a simple test environment - open space for easy navigation."""
    grid_size = (int(size[0] / resolution), int(size[1] / resolution))
    occupancy = np.ones(grid_size, dtype=np.float32)
    
    # Create border walls only
    border_thickness = 3
    occupancy[0:border_thickness, :] = 0  # Left wall
    occupancy[-border_thickness:, :] = 0  # Right wall
    occupancy[:, 0:border_thickness] = 0  # Bottom wall
    occupancy[:, -border_thickness:] = 0  # Top wall
    
    # Add a few small obstacles for interest (but not blocking paths)
    # Small obstacle in one corner
    obs_size = 5
    occupancy[int(grid_size[0]*0.7):int(grid_size[0]*0.7)+obs_size, 
              int(grid_size[1]*0.7):int(grid_size[1]*0.7)+obs_size] = 0
    
    # Another small obstacle
    occupancy[int(grid_size[0]*0.3):int(grid_size[0]*0.3)+obs_size, 
              int(grid_size[1]*0.3):int(grid_size[1]*0.3)+obs_size] = 0
    
    return occupancy


def main():
    print("=" * 60)
    print("Fast Greedy Search Demonstration")
    print("=" * 60)
    
    # Parameters
    map_size = (10.0, 10.0)
    resolution = 0.1
    dt = 0.1
    max_steps = 1000
    
    # Create environment
    print("\n1. Creating environment...")
    env = Environment(map_image_path=None, map_size=map_size, resolution=resolution)
    env.occupancy_map = create_simple_map(map_size, resolution)
    
    # Add objectives randomly
    env.add_random_objectives(num_objectives=3, obj_types=['person', 'pet'])
    print(f"   Added {len(env.objectives)} objectives:")
    for i, obj in enumerate(env.objectives):
        print(f"   - {obj.obj_type} at ({obj.position[0]:.2f}, {obj.position[1]:.2f})")
    
    # Create robot - start in center away from walls
    print("\n2. Initializing robot...")
    initial_pos = np.array([5.0, 5.0])  # Center of 10x10 map
    robot = Robot(robot_id=0, initial_position=initial_pos, initial_heading=0.0)
    print(f"   Robot starts at ({initial_pos[0]:.2f}, {initial_pos[1]:.2f})")
    
    # Create SLAM system
    print("\n3. Initializing SLAM...")
    slam = VisualSLAM(map_size=map_size, resolution=resolution, num_particles=200)
    
    # Create vision system
    print("\n4. Initializing computer vision...")
    detector = ObjectDetector(objective_types=['person', 'pet'])
    detection_filter = DetectionParticleFilter(map_size=map_size, num_particles=200)
    
    for obj_type in ['person', 'pet']:
        detection_filter.initialize_objective(obj_type)
    
    # Create GREEDY controller (much faster!)
    print("\n5. Initializing greedy controller...")
    controller = GreedyController(map_size=map_size)
    info_dist = InformationDistribution(grid_size=(50, 50))
    
    # Storage for metrics
    metrics = {
        'time': [],
        'exploration': [],
        'detections': [],
        'distance': []
    }
    
    print("\n6. Running search simulation...")
    print("   (Using greedy controller for speed...)")
    
    # Main simulation loop
    for step in range(max_steps):
        # Sense environment
        robot.sense(env)
        
        # SLAM update
        lidar_angles = np.linspace(0, 2*np.pi, robot.config.lidar_rays, endpoint=False)
        slam.update(robot.control, robot.lidar_data, lidar_angles, dt)
        
        # Computer vision detection
        detections = detector.detect(robot.camera_data, robot.position, env.current_time)
        
        # Update detection filter
        for detection in detections:
            detection_filter.update(detection)
            
            # Check if objective discovered
            for obj in env.objectives:
                if (not obj.discovered and 
                    np.linalg.norm(obj.position - detection.position) < 0.5 and
                    detector.is_confirmed(detection)):
                    obj.discovered = True
                    obj.discovered_time = env.current_time
                    print(f"\n   Discovered {obj.obj_type} at t={env.current_time:.1f}s")
        
        # Build information distribution
        from scipy.ndimage import zoom
        
        unexplored = slam.map.get_unexplored_regions()
        zoom_factor = (50 / unexplored.shape[0], 50 / unexplored.shape[1])
        unexplored_resized = zoom(unexplored, zoom_factor, order=1)
        
        # Feature density
        feature_density = np.abs(np.gradient(slam.map.occupancy)[0]) + \
                         np.abs(np.gradient(slam.map.occupancy)[1])
        feature_density = zoom(feature_density, zoom_factor, order=1)
        feature_density /= (feature_density.sum() + 1e-10)
        
        # Detection distribution
        detection_dist = detection_filter.get_all_distributions(grid_size=(50, 50))
        
        # Adapt weights based on exploration
        exploration_pct = slam.map.get_exploration_percentage()
        info_dist.adapt_weights(exploration_pct)
        
        # Combine distributions
        target_dist = info_dist.compute(unexplored_resized, feature_density, detection_dist)
        
        # GREEDY control - much faster!
        robot_state = np.array([robot.position[0], robot.position[1], robot.heading])
        control = controller.get_next_control(robot_state, target_dist)
        
        robot.set_control(control[0], control[1])
        
        # Step simulation
        robot.step(dt, env)
        env.step(dt)
        
        # Record metrics
        if step % 10 == 0:
            metrics['time'].append(env.current_time)
            metrics['exploration'].append(exploration_pct)
            metrics['detections'].append(sum([obj.discovered for obj in env.objectives]))
            traj = robot.get_trajectory()
            if len(traj) > 1:
                dist = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                metrics['distance'].append(dist)
            
            # Print progress
            if step % 100 == 0:
                print(f"   Step {step}/{max_steps}: Explored {exploration_pct:.1f}%, "
                      f"Found {metrics['detections'][-1]}/{len(env.objectives)} objectives")
        
        # Stop if all objectives found
        if all([obj.discovered for obj in env.objectives]):
            print(f"\n   All objectives found at t={env.current_time:.1f}s!")
            break
    
    print("\n7. Generating visualization...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Environment with trajectory
    ax = axes[0, 0]
    plt.sca(ax)
    
    # Plot environment manually on this subplot
    ax.imshow(env.occupancy_map.T, cmap='gray', origin='lower',
             extent=[0, map_size[0], 0, map_size[1]])
    
    # Show objectives
    for obj in env.objectives:
        color = 'red' if obj.obj_type == 'person' else 'blue'
        marker = 'X' if obj.discovered else 'o'
        ax.plot(obj.position[0], obj.position[1], 
                marker=marker, markersize=10, color=color,
                markeredgecolor='white', markeredgewidth=2)
    
    # Show robot trajectory
    if len(robot.trajectory) > 0:
        traj_array = np.array(robot.trajectory)
        ax.plot(traj_array[:, 0], traj_array[:, 1], 
                '-', color='cyan', alpha=0.7, linewidth=2, label='Trajectory')
    
    # Show current robot position
    ax.plot(robot.position[0], robot.position[1], 'o', markersize=15, 
            color='lime', markeredgecolor='black', markeredgewidth=2, label='Robot')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Environment and Robot Trajectory (Greedy)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')
    ax.set_xlim(0, map_size[0])
    ax.set_ylim(0, map_size[1])
    
    # SLAM map
    ax = axes[0, 1]
    plt.sca(ax)
    plt.imshow(slam.map.occupancy.T, cmap='gray', origin='lower',
              extent=[0, map_size[0], 0, map_size[1]])
    plt.plot(robot.trajectory[-1][0], robot.trajectory[-1][1], 'ro', markersize=10)
    plt.title(f"SLAM Map ({slam.map.get_exploration_percentage():.1f}% explored)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.colorbar(label="Occupancy Probability")
    
    # Metrics over time
    ax = axes[1, 0]
    plt.sca(ax)
    ax_twin = ax.twinx()
    ax.plot(metrics['time'], metrics['exploration'], 'b-', linewidth=2, label='Exploration %')
    ax_twin.plot(metrics['time'], metrics['detections'], 'r-', linewidth=2, label='Detections')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration %', color='b')
    ax_twin.set_ylabel('Objectives Found', color='r')
    ax.set_title('Search Progress')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')
    
    # Distance traveled
    ax = axes[1, 1]
    plt.sca(ax)
    if len(metrics['distance']) > 0:
        plt.plot(metrics['time'][:len(metrics['distance'])], 
                metrics['distance'], 'g-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance Traveled (m)')
        plt.title('Total Distance Traveled')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 
                               'greedy_demo.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved visualization to: {output_path}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total time: {env.current_time:.1f} seconds")
    print(f"Final exploration: {slam.map.get_exploration_percentage():.1f}%")
    print(f"Objectives found: {sum([obj.discovered for obj in env.objectives])}/{len(env.objectives)}")
    print(f"Total distance traveled: {np.sum(np.linalg.norm(np.diff(robot.get_trajectory(), axis=0), axis=1)):.2f} m")
    
    for obj in env.objectives:
        if obj.discovered:
            print(f"  - {obj.obj_type} found at t={obj.discovered_time:.1f}s")
        else:
            print(f"  - {obj.obj_type} NOT FOUND")
    
    print("\nDemonstration complete!")
    print(f"  View results in: {output_path}")
    print("\nNote: This uses a greedy controller (baseline) instead of ergodic control")
    print("      for faster execution. The robot moves toward highest probability regions.")
    
    plt.show()


if __name__ == "__main__":
    main()
