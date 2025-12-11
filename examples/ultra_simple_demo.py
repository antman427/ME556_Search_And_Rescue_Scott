"""
Ultra-simple demo with NO obstacles - just open space.
Guaranteed to work without getting stuck!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments.environment import Environment
from src.environments.robot import Robot
from src.vision.detector import ObjectDetector
from src.utils.baselines import GreedyController


def main():
    print("=" * 60)
    print("Ultra-Simple Demo - Open Environment")
    print("=" * 60)
    
    # Parameters
    map_size = (10.0, 10.0)
    dt = 0.1
    max_steps = 800  # Increased to allow tighter spiral to cover area
    
    # Create EMPTY environment (no obstacles!)
    print("\n1. Creating open environment (no obstacles)...")
    env = Environment(map_image_path=None, map_size=map_size, resolution=0.1)
    # env.occupancy_map is already all ones (free space)
    
    # Add objectives in known locations for testing
    print("\n2. Adding objectives...")
    env.add_objective(np.array([3.0, 3.0]), 'person')
    env.add_objective(np.array([7.0, 3.0]), 'pet')
    env.add_objective(np.array([5.0, 7.0]), 'person')
    
    print(f"   Added {len(env.objectives)} objectives:")
    for obj in env.objectives:
        print(f"   - {obj.obj_type} at ({obj.position[0]:.2f}, {obj.position[1]:.2f})")
    
    # Create robot in center
    print("\n3. Initializing robot...")
    robot = Robot(robot_id=0, initial_position=np.array([5.0, 5.0]), initial_heading=0.0)
    
    # Create vision system
    print("\n4. Initializing detector...")
    detector = ObjectDetector(objective_types=['person', 'pet'])
    
    # Simple controller - just spiral outward
    print("\n5. Using simple spiral controller...")
    
    print("\n6. Running search...")
    
    found_count = 0
    for step in range(max_steps):
        # Sense
        robot.sense(env)
        
        # Detect
        detections = detector.detect(robot.camera_data, robot.position, env.current_time)
        
        for detection in detections:
            for obj in env.objectives:
                if (not obj.discovered and 
                    np.linalg.norm(obj.position - detection.position) < 0.5 and
                    detector.is_confirmed(detection)):
                    obj.discovered = True
                    obj.discovered_time = env.current_time
                    found_count += 1
                    print(f"   Found {obj.obj_type} at t={env.current_time:.1f}s ({found_count}/{len(env.objectives)})")
        
        # Simple spiral control - TIGHTER SPIRAL
        v = 0.4  # Forward speed (reduced for tighter spiral)
        omega = 0.8  # Higher turn rate for tighter spiral (was 0.3)
        
        robot.set_control(v, omega)
        robot.step(dt, env)
        env.step(dt)
        
        # Stop if all found
        if all([obj.discovered for obj in env.objectives]):
            print(f"\n   All objectives found!")
            break
    
    print("\n7. Creating visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # Environment and trajectory
    plt.subplot(2, 2, 1)
    # Plot map manually
    plt.imshow(np.ones((100, 100)), cmap='gray', origin='lower',
              extent=[0, map_size[0], 0, map_size[1]], vmin=0, vmax=1)
    
    # Plot objectives
    for obj in env.objectives:
        color = 'red' if obj.obj_type == 'person' else 'blue'
        marker = 'X' if obj.discovered else 'o'
        plt.plot(obj.position[0], obj.position[1], 
                marker=marker, markersize=12, color=color,
                markeredgecolor='white', markeredgewidth=2,
                label=f'{obj.obj_type} {"(found)" if obj.discovered else ""}')
    
    # Plot trajectory
    traj = np.array(robot.trajectory)
    plt.plot(traj[:, 0], traj[:, 1], 'g-', linewidth=2, alpha=0.7, label='Robot path')
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'r^', markersize=12, label='End')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Spiral Search Pattern')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, map_size[0])
    plt.ylim(0, map_size[1])
    plt.axis('equal')
    
    # Trajectory detail
    plt.subplot(2, 2, 2)
    traj = np.array(robot.trajectory)
    plt.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, alpha=0.7)
    plt.plot(traj[0, 0], traj[0, 1], 'go', markersize=15, label='Start')
    plt.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=15, label='End')
    
    for obj in env.objectives:
        color = 'red' if obj.obj_type == 'person' else 'blue'
        marker = 'X' if obj.discovered else 'o'
        plt.plot(obj.position[0], obj.position[1], 
                marker=marker, markersize=12, color=color,
                markeredgecolor='white', markeredgewidth=2)
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Trajectory Details')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Distance over time
    plt.subplot(2, 2, 3)
    distances = [0]
    for i in range(1, len(traj)):
        distances.append(distances[-1] + np.linalg.norm(traj[i] - traj[i-1]))
    times = np.arange(len(distances)) * dt
    plt.plot(times, distances, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Distance Traveled (m)')
    plt.title('Distance Over Time')
    plt.grid(True, alpha=0.3)
    
    # Summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""
    RESULTS SUMMARY
    
    Total Time: {env.current_time:.1f} seconds
    
    Objectives Found: {found_count}/{len(env.objectives)}
    
    Distance Traveled: {distances[-1]:.2f} m
    
    Search Pattern: Spiral
    
    Environment: Open (no obstacles)
    
    """
    
    for i, obj in enumerate(env.objectives):
        if obj.discovered:
            summary_text += f"\n  {obj.obj_type} #{i+1}: Found at {obj.discovered_time:.1f}s"
        else:
            summary_text += f"\n  {obj.obj_type} #{i+1}: NOT FOUND"
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 
                               'ultra_simple_demo.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Ultra-simple demo complete!")
    print(f"  Found {found_count}/{len(env.objectives)} objectives")
    print(f"  Distance: {distances[-1]:.2f} m")
    print("=" * 60)
    
    plt.show()


if __name__ == "__main__":
    main()
