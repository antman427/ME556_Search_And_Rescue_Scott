"""
Environment Generators for Different Scenario Types

Creates realistic maps for:
1. Indoor (office building with rooms and hallways)
2. Disaster (collapsed building with rubble)
3. Outdoor (open area with trees and structures)
4. Mixed (combination environment)

Each environment tests system generalization and robustness.
"""

import numpy as np
from typing import Tuple
from scipy.ndimage import binary_dilation, gaussian_filter


def generate_indoor_office(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Generate indoor office environment with rooms and hallways.
    
    Args:
        size: Grid size (height, width)
        
    Returns:
        Occupancy map (1 = free, 0 = occupied)
    """
    h, w = size
    occupancy = np.ones((h, w), dtype=np.float32)
    
    # Outer walls
    border = 3
    occupancy[0:border, :] = 0
    occupancy[-border:, :] = 0
    occupancy[:, 0:border] = 0
    occupancy[:, -border:] = 0
    
    # Horizontal hallway (middle)
    hallway_width = 8
    hallway_y = h // 2
    
    # Vertical hallway (middle)
    hallway_x = w // 2
    
    # Create rooms by adding walls
    wall_thickness = 2
    
    # Top-left room
    occupancy[h//4-wall_thickness:h//4+wall_thickness, border:hallway_x-hallway_width//2] = 0
    occupancy[border:hallway_y-hallway_width//2, w//4-wall_thickness:w//4+wall_thickness] = 0
    
    # Top-right room
    occupancy[h//4-wall_thickness:h//4+wall_thickness, hallway_x+hallway_width//2:w-border] = 0
    occupancy[border:hallway_y-hallway_width//2, 3*w//4-wall_thickness:3*w//4+wall_thickness] = 0
    
    # Bottom-left room
    occupancy[3*h//4-wall_thickness:3*h//4+wall_thickness, border:hallway_x-hallway_width//2] = 0
    occupancy[hallway_y+hallway_width//2:h-border, w//4-wall_thickness:w//4+wall_thickness] = 0
    
    # Bottom-right room
    occupancy[3*h//4-wall_thickness:3*h//4+wall_thickness, hallway_x+hallway_width//2:w-border] = 0
    occupancy[hallway_y+hallway_width//2:h-border, 3*w//4-wall_thickness:3*w//4+wall_thickness] = 0
    
    # Add doorways (gaps in walls)
    door_width = 6
    
    # Doors to center
    occupancy[hallway_y-door_width//2:hallway_y+door_width//2, 
              hallway_x-wall_thickness:hallway_x+wall_thickness] = 1
    occupancy[h//4-wall_thickness:h//4+wall_thickness,
              hallway_x-door_width//2:hallway_x+door_width//2] = 1
    occupancy[3*h//4-wall_thickness:3*h//4+wall_thickness,
              hallway_x-door_width//2:hallway_x+door_width//2] = 1
    
    # Add some furniture/obstacles in rooms
    np.random.seed(42)
    for _ in range(8):
        room_x = np.random.randint(border+5, w-border-5)
        room_y = np.random.randint(border+5, h-border-5)
        # Small rectangular obstacle
        occupancy[room_y:room_y+3, room_x:room_x+4] = 0
    
    return occupancy


def generate_disaster_rubble(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Generate disaster/collapsed building environment with irregular rubble.
    
    Args:
        size: Grid size (height, width)
        
    Returns:
        Occupancy map (1 = free, 0 = occupied)
    """
    h, w = size
    occupancy = np.ones((h, w), dtype=np.float32)
    
    # Outer walls
    border = 3
    occupancy[0:border, :] = 0
    occupancy[-border:, :] = 0
    occupancy[:, 0:border] = 0
    occupancy[:, -border:] = 0
    
    # Create partial walls (collapsed structure)
    np.random.seed(42)
    
    # Partial walls at various angles
    for i in range(5):
        wall_start_x = np.random.randint(border+10, w-border-10)
        wall_start_y = np.random.randint(border+10, h-border-10)
        wall_length = np.random.randint(15, 35)
        wall_angle = np.random.uniform(0, np.pi)
        wall_thickness = np.random.randint(2, 4)
        
        for j in range(wall_length):
            x = int(wall_start_x + j * np.cos(wall_angle))
            y = int(wall_start_y + j * np.sin(wall_angle))
            
            if border < x < w-border and border < y < h-border:
                occupancy[y-wall_thickness:y+wall_thickness, 
                         x-wall_thickness:x+wall_thickness] = 0
    
    # Add irregular rubble piles
    for _ in range(15):
        rubble_x = np.random.randint(border+5, w-border-5)
        rubble_y = np.random.randint(border+5, h-border-5)
        rubble_size = np.random.randint(3, 8)
        
        # Create irregular shape
        rubble_shape = np.random.rand(rubble_size*2, rubble_size*2) > 0.5
        rubble_shape = binary_dilation(rubble_shape, iterations=2).astype(float)
        
        y_start = max(border, rubble_y - rubble_size)
        y_end = min(h-border, rubble_y + rubble_size)
        x_start = max(border, rubble_x - rubble_size)
        x_end = min(w-border, rubble_x + rubble_size)
        
        rubble_h = y_end - y_start
        rubble_w = x_end - x_start
        
        if rubble_h > 0 and rubble_w > 0:
            rubble_resized = rubble_shape[:rubble_h, :rubble_w]
            occupancy[y_start:y_end, x_start:x_end] = np.where(
                rubble_resized > 0.5, 0, occupancy[y_start:y_end, x_start:x_end]
            )
    
    # Add some larger debris
    for _ in range(6):
        debris_x = np.random.randint(border+10, w-border-10)
        debris_y = np.random.randint(border+10, h-border-10)
        debris_w = np.random.randint(8, 15)
        debris_h = np.random.randint(8, 15)
        
        occupancy[debris_y:debris_y+debris_h, debris_x:debris_x+debris_w] = 0
    
    return occupancy


def generate_outdoor_park(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Generate outdoor park/wilderness environment with trees and structures.
    
    Args:
        size: Grid size (height, width)
        
    Returns:
        Occupancy map (1 = free, 0 = occupied)
    """
    h, w = size
    occupancy = np.ones((h, w), dtype=np.float32)
    
    # Boundary (fence/natural border)
    border = 3
    occupancy[0:border, :] = 0
    occupancy[-border:, :] = 0
    occupancy[:, 0:border] = 0
    occupancy[:, -border:] = 0
    
    # Add trees (circular obstacles scattered around)
    np.random.seed(42)
    num_trees = 25
    
    for _ in range(num_trees):
        tree_x = np.random.randint(border+5, w-border-5)
        tree_y = np.random.randint(border+5, h-border-5)
        tree_radius = np.random.randint(2, 4)
        
        # Create circular tree
        for dy in range(-tree_radius, tree_radius+1):
            for dx in range(-tree_radius, tree_radius+1):
                if dx*dx + dy*dy <= tree_radius*tree_radius:
                    y = tree_y + dy
                    x = tree_x + dx
                    if 0 <= y < h and 0 <= x < w:
                        occupancy[y, x] = 0
    
    # Add some bushes/shrubs (smaller obstacles in clusters)
    for _ in range(8):
        cluster_x = np.random.randint(border+10, w-border-10)
        cluster_y = np.random.randint(border+10, h-border-10)
        
        # Multiple small obstacles
        for _ in range(np.random.randint(3, 6)):
            bush_x = cluster_x + np.random.randint(-5, 5)
            bush_y = cluster_y + np.random.randint(-5, 5)
            bush_size = np.random.randint(1, 3)
            
            if (border < bush_x < w-border and border < bush_y < h-border):
                occupancy[bush_y-bush_size:bush_y+bush_size+1,
                         bush_x-bush_size:bush_x+bush_size+1] = 0
    
    # Add a few structures (buildings/shelters)
    for _ in range(3):
        struct_x = np.random.randint(border+10, w-border-20)
        struct_y = np.random.randint(border+10, h-border-20)
        struct_w = np.random.randint(8, 15)
        struct_h = np.random.randint(8, 15)
        
        # Hollow structure (walls with interior space)
        wall_thickness = 2
        occupancy[struct_y:struct_y+wall_thickness, struct_x:struct_x+struct_w] = 0
        occupancy[struct_y+struct_h-wall_thickness:struct_y+struct_h, struct_x:struct_x+struct_w] = 0
        occupancy[struct_y:struct_y+struct_h, struct_x:struct_x+wall_thickness] = 0
        occupancy[struct_y:struct_y+struct_h, struct_x+struct_w-wall_thickness:struct_x+struct_w] = 0
        
        # Add door
        door_pos = struct_x + struct_w // 2
        occupancy[struct_y:struct_y+wall_thickness, door_pos-2:door_pos+2] = 1
    
    # Add paths (clear corridors)
    path_width = 5
    # Horizontal path
    occupancy[h//2-path_width:h//2+path_width, border:w-border] = 1
    # Vertical path
    occupancy[border:h-border, w//2-path_width:w//2+path_width] = 1
    
    return occupancy


def generate_mixed_environment(size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Generate mixed environment combining indoor, disaster, and outdoor elements.
    
    Args:
        size: Grid size (height, width)
        
    Returns:
        Occupancy map (1 = free, 0 = occupied)
    """
    h, w = size
    
    # Start with outdoor base
    occupancy = generate_outdoor_park(size)
    
    # Add indoor section (top-left)
    indoor_section = generate_indoor_office((h//2, w//2))
    occupancy[0:h//2, 0:w//2] = indoor_section
    
    # Add disaster section (bottom-right)
    disaster_section = generate_disaster_rubble((h//2, w//2))
    occupancy[h//2:h, w//2:w] = disaster_section
    
    # Smooth transitions between sections
    transition_zone = 10
    for i in range(transition_zone):
        weight = i / transition_zone
        # Vertical transition
        occupancy[h//2-transition_zone//2+i, :] = np.where(
            occupancy[h//2-transition_zone//2+i, :] > 0,
            1.0,
            occupancy[h//2-transition_zone//2+i, :]
        )
        # Horizontal transition
        occupancy[:, w//2-transition_zone//2+i] = np.where(
            occupancy[:, w//2-transition_zone//2+i] > 0,
            1.0,
            occupancy[:, w//2-transition_zone//2+i]
        )
    
    return occupancy


def get_safe_spawn_positions(occupancy: np.ndarray, num_positions: int,
                             map_size: Tuple[float, float],
                             min_separation: float = 2.0) -> list:
    """
    Find safe spawn positions in free space with minimum separation.
    
    Args:
        occupancy: Occupancy map
        num_positions: Number of positions needed
        map_size: Physical size in meters
        min_separation: Minimum distance between positions
        
    Returns:
        List of (x, y) positions in meters
    """
    h, w = occupancy.shape
    positions = []
    
    # Create distance transform to stay away from walls
    from scipy.ndimage import distance_transform_edt
    distance_map = distance_transform_edt(occupancy)
    
    # Find good positions (far from walls and each other)
    max_attempts = 1000
    for _ in range(max_attempts):
        if len(positions) >= num_positions:
            break
        
        # Sample from areas far from walls
        good_cells = distance_map > 5  # At least 5 cells from walls
        if not good_cells.any():
            good_cells = occupancy > 0.5
        
        candidates = np.argwhere(good_cells)
        if len(candidates) == 0:
            continue
        
        idx = np.random.choice(len(candidates))
        grid_y, grid_x = candidates[idx]
        
        # Convert to world coordinates
        x = (grid_x / w) * map_size[0]
        y = (grid_y / h) * map_size[1]
        pos = np.array([x, y])
        
        # Check separation from existing positions
        if len(positions) == 0:
            positions.append(pos)
        else:
            min_dist = min([np.linalg.norm(pos - p) for p in positions])
            if min_dist > min_separation:
                positions.append(pos)
    
    return positions


def get_safe_objective_positions(occupancy: np.ndarray, num_objectives: int,
                                map_size: Tuple[float, float],
                                min_separation: float = 1.5) -> list:
    """
    Find safe objective positions in free space.
    
    Args:
        occupancy: Occupancy map
        num_objectives: Number of objectives
        map_size: Physical size in meters
        min_separation: Minimum distance between objectives
        
    Returns:
        List of (x, y) positions in meters
    """
    return get_safe_spawn_positions(occupancy, num_objectives, map_size, min_separation)


# Export all generators
ENVIRONMENT_TYPES = {
    'indoor': generate_indoor_office,
    'disaster': generate_disaster_rubble,
    'outdoor': generate_outdoor_park,
    'mixed': generate_mixed_environment
}
