"""
Multi-Robot Cooperative Search Demonstration

Compares 1 robot vs 2 robots vs 3 robots on the same search task.
Shows the advantage of multi-robot coordination.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.environment import Environment
from src.environments.robot import Robot
from src.slam.slam import VisualSLAM
from src.vision.detector import ObjectDetector, DetectionParticleFilter
from src.ergodic.camera_aware_controller import CameraAwareErgodicController
from src.ergodic.controller import InformationDistribution
from src.coordination.coordinator import MultiRobotCoordinator
from scipy.ndimage import zoom


def create_simple_map(size=(10, 10), resolution=0.1):
    """Create test environment."""
    grid_size = (int(size[0] / resolution), int(size[1] / resolution))
    occupancy = np.ones(grid_size, dtype=np.float32)
    
    border_thickness = 3
    occupancy[0:border_thickness, :] = 0
    occupancy[-border_thickness:, :] = 0
    occupancy[:, 0:border_thickness] = 0
    occupancy[:, -border_thickness:] = 0
    
    obs_size = 5
    occupancy[int(grid_size[0]*0.7):int(grid_size[0]*0.7)+obs_size, 
              int(grid_size[1]*0.7):int(grid_size[1]*0.7)+obs_size] = 0
    
    occupancy[int(grid_size[0]*0.3):int(grid_size[0]*0.3)+obs_size, 
              int(grid_size[1]*0.3):int(grid_size[1]*0.3)+obs_size] = 0
    
    return occupancy


def run_single_robot(env_base, objectives_base, map_size, resolution, dt, max_steps):
    """Run single robot search (baseline)."""
    # Create environment copy
    env = Environment(map_image_path=None, map_size=map_size, resolution=resolution)
    env.occupancy_map = env_base.occupancy_map.copy()
    
    for obj_base in objectives_base:
        env.add_objective(obj_base.position.copy(), obj_base.obj_type, obj_base.size)
    
    # Create robot
    robot = Robot(robot_id=0, initial_position=np.array([5.0, 5.0]), initial_heading=0.0)
    slam = VisualSLAM(map_size=map_size, resolution=resolution, num_particles=200)
    detector = ObjectDetector(objective_types=['person', 'pet'])
    detection_filter = DetectionParticleFilter(map_size=map_size, num_particles=200)
    for obj_type in ['person', 'pet']:
        detection_filter.initialize_objective(obj_type)
    
    controller = CameraAwareErgodicController(
        map_size=map_size, num_basis=3, horizon=15, dt=dt,
        camera_fov=robot.config.camera_fov
    )
    info_dist = InformationDistribution(grid_size=(50, 50))
    
    # Run simulation
    for step in range(max_steps):
        robot.sense(env)
        
        lidar_angles = np.linspace(0, 2*np.pi, robot.config.lidar_rays, endpoint=False)
        slam.update(robot.control, robot.lidar_data, lidar_angles, dt)
        
        detections = detector.detect(robot.camera_data, robot.position, env.current_time)
        
        for detection in detections:
            detection_filter.update(detection)
            
            for obj in env.objectives:
                if (not obj.discovered and 
                    np.linalg.norm(obj.position - detection.position) < 0.5 and
                    detector.is_confirmed(detection)):
                    obj.discovered = True
                    obj.discovered_time = env.current_time
                    print(f"      Robot 0 found {obj.obj_type} at t={env.current_time:.1f}s")
        
        unexplored = slam.map.get_unexplored_regions()
        zoom_factor = (50 / unexplored.shape[0], 50 / unexplored.shape[1])
        unexplored_resized = zoom(unexplored, zoom_factor, order=1)
        
        feature_density = np.abs(np.gradient(slam.map.occupancy)[0]) + \
                         np.abs(np.gradient(slam.map.occupancy)[1])
        feature_density = zoom(feature_density, zoom_factor, order=1)
        feature_density /= (feature_density.sum() + 1e-10)
        
        detection_dist = detection_filter.get_all_distributions(grid_size=(50, 50))
        
        exploration_pct = slam.map.get_exploration_percentage()
        info_dist.adapt_weights(exploration_pct)
        
        target_dist = info_dist.compute(unexplored_resized, feature_density, detection_dist)
        
        robot_state = np.array([robot.position[0], robot.position[1], robot.heading])
        control = controller.get_next_control(robot_state, target_dist)
        
        robot.set_control(control[0], control[1])
        robot.step(dt, env)
        env.step(dt)
        
        if step % 50 == 0 and step > 0:
            found = sum([obj.discovered for obj in env.objectives])
            print(f"      Step {step}/{max_steps}: Explored {exploration_pct:.1f}%, Found {found}/{len(env.objectives)}")
        
        if all([obj.discovered for obj in env.objectives]):
            print(f"      All objectives found at t={env.current_time:.1f}s!")
            break
    
    return {
        'robot': robot,
        'slam': slam,
        'env': env,
        'time': env.current_time,
        'exploration': slam.map.get_exploration_percentage(),
        'found': sum([obj.discovered for obj in env.objectives])
    }


def run_multi_robot(num_robots, env_base, objectives_base, map_size, resolution, dt, max_steps):
    """Run multi-robot coordinated search."""
    # Create environment copy
    env = Environment(map_image_path=None, map_size=map_size, resolution=resolution)
    env.occupancy_map = env_base.occupancy_map.copy()
    
    for obj_base in objectives_base:
        env.add_objective(obj_base.position.copy(), obj_base.obj_type, obj_base.size)
    
    # Spawn positions (spread around map edges to avoid initial clustering)
    if num_robots == 2:
        spawn_positions = [
            np.array([2.0, 2.0]),  # Bottom-left quadrant
            np.array([8.0, 8.0])   # Top-right quadrant (far apart)
        ]
    elif num_robots == 3:
        spawn_positions = [
            np.array([2.5, 5.0]),
            np.array([5.0, 8.0]),
            np.array([7.5, 2.0])
        ]
    else:
        spawn_positions = [np.array([5.0, 5.0]) for _ in range(num_robots)]
    
    # Create coordinator
    coordinator = MultiRobotCoordinator(
        num_robots=num_robots,
        environment=env,
        spawn_positions=spawn_positions,
        map_size=map_size,
        resolution=resolution,
        dt=dt
    )
    
    # Run simulation
    for step in range(max_steps):
        coordinator.step()
        
        if step % 50 == 0 and step > 0:
            results = coordinator.get_results()
            print(f"      Step {step}/{max_steps}: Explored {results['exploration']:.1f}%, "
                  f"Found {results['objectives_found']}/{results['total_objectives']}")
        
        if coordinator.all_objectives_found():
            results = coordinator.get_results()
            print(f"      All objectives found at t={results['time']:.1f}s!")
            break
    
    return coordinator.get_results()


def main():
    print("=" * 70)
    print("MULTI-ROBOT COOPERATIVE SEARCH DEMONSTRATION")
    print("=" * 70)
    
    # Parameters
    map_size = (10.0, 10.0)
    resolution = 0.1
    dt = 0.1
    max_steps = 500
    
    # Create base environment (same for all)
    print("\n1. Creating test environment...")
    env_base = Environment(map_image_path=None, map_size=map_size, resolution=resolution)
    env_base.occupancy_map = create_simple_map(map_size, resolution)
    
    # Add objectives at fixed locations
    print("   Adding 3 objectives...")
    objectives_base = []
    for pos, obj_type in [
        (np.array([2.0, 3.5]), 'person'),
        (np.array([8.0, 6.5]), 'pet'),
        (np.array([4.5, 8.5]), 'person')
    ]:
        obj = type('Objective', (), {
            'position': pos,
            'obj_type': obj_type,
            'size': 0.3,
            'discovered': False,
            'discovered_time': None
        })()
        objectives_base.append(obj)
        print(f"      - {obj_type} at ({pos[0]:.1f}, {pos[1]:.1f})")
    
    # Run 1 robot
    print("\n2. Running 1 ROBOT search...")
    results_1 = run_single_robot(env_base, objectives_base, map_size, resolution, dt, max_steps)
    
    # Run 2 robots
    print("\n3. Running 2 ROBOTS coordinated search...")
    results_2 = run_multi_robot(2, env_base, objectives_base, map_size, resolution, dt, max_steps)
    
    # Run 3 robots
    print("\n4. Running 3 ROBOTS coordinated search...")
    results_3 = run_multi_robot(3, env_base, objectives_base, map_size, resolution, dt, max_steps)
    
    # Create visualization
    print("\n5. Creating comparison visualization...")
    
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Multi-Robot Cooperative Search: 1 vs 2 vs 3 Robots', 
                 fontsize=16, fontweight='bold')
    
    results_list = [results_1, results_2, results_3]
    titles = ['1 ROBOT', '2 ROBOTS', '3 ROBOTS']
    colors_list = [['cyan'], ['red', 'blue'], ['red', 'green', 'blue']]
    
    for col, (results, title, colors) in enumerate(zip(results_list, titles, colors_list)):
        
        # Row 1: Trajectories
        ax = plt.subplot(3, 3, col + 1)
        
        if 'robot' in results:
            # Single robot
            env = results['env']
            robot = results['robot']
            
            ax.imshow(env.occupancy_map.T, cmap='gray', origin='lower',
                     extent=[0, map_size[0], 0, map_size[1]])
            
            traj = np.array(robot.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[0], 
                   linewidth=2, alpha=0.7, label='Robot 0')
            ax.plot(robot.position[0], robot.position[1], 'o', 
                   markersize=12, color=colors[0], markeredgecolor='black', markeredgewidth=2)
        else:
            # Multi-robot
            env = results['objectives'][0].__class__.__module__  # Get env reference
            ax.imshow(env_base.occupancy_map.T, cmap='gray', origin='lower',
                     extent=[0, map_size[0], 0, map_size[1]])
            
            for i, robot in enumerate(results['robots']):
                traj = np.array(robot.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[i], 
                       linewidth=2, alpha=0.7, label=f'Robot {i}')
                ax.plot(robot.position[0], robot.position[1], 'o', 
                       markersize=12, color=colors[i], markeredgecolor='black', markeredgewidth=2)
                
                # Draw circle showing camera range
                circle = Circle(robot.position, robot.config.camera_range, 
                              fill=False, edgecolor=colors[i], linestyle='--', alpha=0.3)
                ax.add_patch(circle)
        
        # Plot objectives
        objs = results['env'].objectives if 'env' in results else results['objectives']
        for obj in objs:
            obj_color = 'red' if obj.obj_type == 'person' else 'blue'
            marker = 'X' if obj.discovered else 'o'
            ax.plot(obj.position[0], obj.position[1], marker=marker, 
                   markersize=15, color=obj_color, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title}\nRobot Trajectories')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, map_size[0])
        ax.set_ylim(0, map_size[1])
        
        # Row 2: SLAM Map with better visualization
        ax = plt.subplot(3, 3, col + 4)
        
        if 'slam' in results:
            slam_map = results['slam'].map
        else:
            slam_map = results['global_map']
        
        # Create custom visualization: unexplored regions highlighted
        # Need shape (height, width, 3) for RGB image
        display_map_rgb = np.zeros((*slam_map.occupancy.T.shape, 3))
        
        # Explored regions: grayscale (black=free, white=occupied)
        explored_map = slam_map.occupancy.T
        display_map_rgb[:, :, 0] = explored_map
        display_map_rgb[:, :, 1] = explored_map
        display_map_rgb[:, :, 2] = explored_map
        
        # Unexplored regions: distinct color (dark blue tint)
        unexplored_mask = ~slam_map.explored.T
        display_map_rgb[unexplored_mask, 0] = 0.2  # R
        display_map_rgb[unexplored_mask, 1] = 0.2  # G
        display_map_rgb[unexplored_mask, 2] = 0.4  # B (blue tint)
        
        ax.imshow(display_map_rgb, origin='lower',
                 extent=[0, map_size[0], 0, map_size[1]])
        
        for obj in objs:
            obj_color = 'red' if obj.obj_type == 'person' else 'yellow'
            marker = 'X' if obj.discovered else 'o'
            ax.plot(obj.position[0], obj.position[1], marker=marker, 
                   markersize=12, color=obj_color, markeredgecolor='white', markeredgewidth=3)
        
        exploration_pct = results['exploration']
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'SLAM Map ({exploration_pct:.1f}% explored)')
        ax.grid(True, alpha=0.3)
        
        # Row 3: Metrics
        ax = plt.subplot(3, 3, col + 7)
        ax.axis('off')
        
        metrics_text = f"PERFORMANCE\n\n"
        metrics_text += f"Number of Robots: {len(colors)}\n"
        metrics_text += f"Exploration: {results['exploration']:.1f}%\n"
        
        found = results['found'] if 'found' in results else results['objectives_found']
        total = 3
        metrics_text += f"Objectives Found: {found}/{total}\n"
        metrics_text += f"Success Rate: {100*found/total:.0f}%\n\n"
        
        metrics_text += f"Time: {results['time']:.1f}s\n"
        
        if 'robot' in results:
            distance = np.sum(np.linalg.norm(np.diff(results['robot'].get_trajectory(), axis=0), axis=1))
            metrics_text += f"Distance: {distance:.1f}m\n"
        else:
            total_dist = 0
            for robot in results['robots']:
                total_dist += np.sum(np.linalg.norm(np.diff(robot.get_trajectory(), axis=0), axis=1))
            metrics_text += f"Total Distance: {total_dist:.1f}m\n"
            metrics_text += f"Avg per Robot: {total_dist/len(colors):.1f}m\n"
        
        metrics_text += f"\nDiscoveries:\n"
        for obj in objs:
            if obj.discovered:
                metrics_text += f"  {obj.obj_type} at {obj.discovered_time:.1f}s\n"
                if hasattr(obj, 'discovered_by'):
                    metrics_text += f"    (by Robot {obj.discovered_by})\n"
            else:
                metrics_text += f"  {obj.obj_type} NOT FOUND\n"
        
        # Color code
        if found == total:
            box_color = 'lightgreen'
        elif found >= total - 1:
            box_color = 'lightyellow'
        else:
            box_color = 'lightcoral'
        
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='center', horizontalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.8, pad=1))
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 
                               'multi_robot_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    print("\n                    1 ROBOT     2 ROBOTS    3 ROBOTS")
    print("-" * 70)
    
    expl_1 = results_1['exploration']
    expl_2 = results_2['exploration']
    expl_3 = results_3['exploration']
    print(f"Exploration:        {expl_1:5.1f}%     {expl_2:5.1f}%     {expl_3:5.1f}%")
    
    found_1 = results_1['found']
    found_2 = results_2['objectives_found']
    found_3 = results_3['objectives_found']
    print(f"Objectives Found:   {found_1:5.0f}/3       {found_2:5.0f}/3       {found_3:5.0f}/3")
    
    print(f"Success Rate:       {100*found_1/3:5.0f}%       {100*found_2/3:5.0f}%       {100*found_3/3:5.0f}%")
    
    time_1 = results_1['time']
    time_2 = results_2['time']
    time_3 = results_3['time']
    print(f"Time:               {time_1:5.1f}s      {time_2:5.1f}s      {time_3:5.1f}s")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    
    print(f"\nAdding robots improved success rate:")
    print(f"  1 robot:  {100*found_1/3:.0f}%")
    print(f"  2 robots: {100*found_2/3:.0f}% ({100*found_2/3 - 100*found_1/3:+.0f}%)")
    print(f"  3 robots: {100*found_3/3:.0f}% ({100*found_3/3 - 100*found_1/3:+.0f}%)")
    
    print(f"\nTime efficiency:")
    if found_2 >= found_1:
        print(f"  2 robots completed in {time_2:.1f}s vs {time_1:.1f}s for 1 robot")
    if found_3 >= found_1:
        print(f"  3 robots completed in {time_3:.1f}s vs {time_1:.1f}s for 1 robot")
    
    print(f"\nWhy multi-robot is better:")
    print("  - Multiple cameras cover more area simultaneously")
    print("  - Distributed search avoids redundant coverage")
    print("  - Shared map accelerates exploration")
    print("  - Different viewpoints increase detection probability")
    
    print("\nComparison complete!")
    print(f"  View results: {output_path}")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    main()
