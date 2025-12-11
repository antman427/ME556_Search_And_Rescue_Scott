# Multi-Robot Visual SLAM and Ergodic Search for General Search and Rescue

A generalizable search and rescue system combining Visual SLAM, computer vision object detection, and ergodic control for autonomous multi-robot objective location in arbitrary unknown environments.

## Project Overview

This system enables 2-3 simulated robots to:
1. Simultaneously build accurate maps using visual SLAM
2. Detect and localize objectives (people, pets, objects) using computer vision
3. Coordinate search patterns using ergodic control

The system accepts any 2D environment map as input and works without prior knowledge of the space or objective locations.

## Key Features

- **Visual SLAM**: Incremental occupancy grid mapping with feature-based localization
- **Computer Vision**: Multi-stage object detection (coarse and fine-grained)
- **Ergodic Control**: Optimal coverage trajectories matching information distributions
- **Multi-Agent Coordination**: Decentralized planning with shared information
- **Generalization**: Works on arbitrary 2D environment maps

## Project Structure

```
ME556_Final_Project_Scott/
├── src/
│   ├── slam/              # SLAM and mapping components
│   ├── vision/            # Computer vision detection
│   ├── ergodic/           # Ergodic control implementation
│   ├── coordination/      # Multi-robot coordination
│   ├── environments/      # Environment simulation
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── configs/               # Configuration files
├── results/               # Output results and visualizations
└── README.md
```

## Installation

### Requirements
- Python 3.8+
- NumPy
- SciPy
- OpenCV (cv2)
- Matplotlib
- FilterPy

### Setup

```bash
python -m pip install numpy scipy opencv-python matplotlib filterpy
```

Clone the ergodic-control-sandbox repository:
```bash
git clone https://github.com/MurpheyLab/ergodic-control-sandbox.git
```
Clone this repository:
```bash
git clone https://github.com/antman427/ME556_Search_And_Rescue_Scott.git
```

## Usage

### Basic Single Robot Example

```python
from src.environments.environment import Environment
from src.slam.slam import VisualSLAM
from src.vision.detector import ObjectDetector
from src.ergodic.controller import ErgodicController

# Load environment
env = Environment()

# Initialize components
slam = VisualSLAM()
detector = ObjectDetector(templates=['person', 'pet'])
controller = ErgodicController()

# Run search
# ... (see examples/)
```

### Multi-Robot Search

```python
from src.coordination.coordinator import MultiRobotCoordinator

coordinator = MultiRobotCoordinator(num_robots=3)
coordinator.run_search(environment, objectives=['person', 'pet'])
```

## Environment Maps

The system accepts any 2D grayscale or RGB image as an environment map:
- Dark pixels (< 128) = obstacles/walls
- Light pixels (>= 128) = navigable space

Example maps provided in `environments/maps/`:
- `office.png` - Structured indoor environment
- `disaster.png` - Collapsed building with irregular obstacles
- `outdoor.png` - Park or field with scattered obstacles
- `mixed.png` - Building interior transitioning to outdoor

---

## Installation

1. **Install Python dependencies:**
```bash
python -m pip install -r requirements.txt
```

Required packages:
- numpy >= 1.21.0
- scipy >= 1.7.0
- opencv-python >= 4.5.0
- matplotlib >= 3.4.0
- filterpy >= 1.4.5

---

## Running Tests

### Quick Component Test
Verify all components work correctly:

```bash
python tests/test_components.py
```

Expected output: All tests should PASS.

---

## Running Examples

### 1. Ultra-Simple Demo (START HERE!)

The easiest way to verify everything works:

```bash
python examples/ultra_simple_demo.py
```

This runs in ~10 seconds with:
- Open environment (no obstacles to get stuck on!)
- Fixed objective locations (for reproducibility)
- Simple spiral search pattern
- **Guaranteed to work!**

### 2. Greedy Controller Demo

Full system with SLAM and intelligent search:

```bash
python examples/greedy_demo.py
```

This will:
1. Create environment with small obstacles
2. Place 3 random objectives
3. Run greedy search with SLAM
4. Generate visualizations
5. Should find 2-3 objectives in 30-60 seconds

### 3. Ergodic Controller Demo

Advanced optimal control (slower):

```bash
python examples/one_robot_ergodic_control.py
```

This:
1. Uses ergodic control for theoretically optimal coverage. Slower but more systematic.
2. Should find 2-3 objectives in 30-60 seconds

---

### 4. Ergodic Controller Muliple Robots

Ergodic Controller with 1, 2, and 3 robots:

```bash
python examples/multi_robot_ergodic_control.py
```

This:
1. Uses ergodic control with coordination from other robots to better improver mapping.
2. Takes longer to run since it runs three iterations (1 Robot, 2 Robots, then 3 Robots)

---
### 5. Muliple Environments

Greedy and Ergodic Controllers with different environments to show the improvement with Ergodic Control:

```bash
python examples/multi_environments_comparsion.py
```

This:
1. Uses ergodic control for theoretically optimal coverage.
2. Was shortened to smaller iterations and only two environments because this version takes less than an hour to run. Other environements exist and you can swap them out.

---

## Understanding the Output

The demo generates a 4-panel visualization:

1. **Environment and Trajectory** (top-left)
   - Shows the environment, obstacles, objectives, and robot path
   - Red X = discovered objectives
   - Red O = undiscovered objectives
   - Colored line = robot trajectory

2. **SLAM Map** (top-right)
   - Shows the map built by the robot using SLAM
   - Gray = obstacles, White = free space
   - Percentage explored shown in title

3. **Search Progress** (bottom-left)
   - Blue line = exploration percentage over time
   - Red line = number of objectives found over time

4. **Ergodic Metric** (bottom-right)
   - Shows how well the trajectory matches target distribution
   - Lower values = better coverage

---

## Key Components

### Environment (`src/environments/environment.py`)
- Loads arbitrary 2D maps
- Simulates LIDAR and camera sensors
- Manages objectives

### Robot (`src/environments/robot.py`)
- Differential drive dynamics
- Sensor models
- Trajectory tracking

### SLAM (`src/slam/slam.py`)
- Occupancy grid mapping
- Particle filter localization
- Ray tracing for map updates

### Vision (`src/vision/detector.py`)
- Object detection with confidence scoring
- Detection particle filter for tracking

### Ergodic Control (`src/ergodic/controller.py`)
- Fourier-based trajectory optimization
- Information distribution management
- Receding horizon planning

---

## Results

Results are saved to `results/` directory:
- Performance plots (PNG)
- Map building visualization
- Output of console, if you don't want to run these

## Course Context

This project is for a Master's level Robotic Concepts course, integrating:
- SLAM (Simultaneous Localization and Mapping)
- Particle Filters
- Ergodic Control
- Multi-Agent Coordination
- Computer Vision

## License

MIT License - Academic Project

## Author

Anthony Scott - Pennsylvania State University (World Campus)
ME556 Robotic Concepts

## References

- Ergodic Control Sandbox: https://github.com/MurpheyLab/ergodic-control-sandbox
- @article{mathew2011metrics,
    title={Metrics for ergodicity and design of ergodic dynamics for multi-agent systems},
    author={Mathew, George and Mezi{\'c}, Igor},
    journal={Physica D: Nonlinear Phenomena},
    volume={240},
    number={4-5},
    pages={432--442},
    year={2011},
    publisher={Elsevier}
    }

- @inproceedings{mathew2009spectral,
    title={Spectral multiscale coverage: A uniform coverage algorithm for mobile sensor networks},
    author={Mathew, George and Mezi{\'c}, Igor},
    booktitle={Proceedings of the 48th IEEE Conference on Decision and Control},
    pages={7872--7877},
    year={2009},
    organization={IEEE}
    }

- @article{miller2016ergodic,
    title={Ergodic exploration of distributed information},
    author={Miller, Lauren M and Silverman, Yonatan and MacIver, Malcolm A and Murphey, Todd D},
    journal={IEEE Transactions on Robotics},
    volume={32},
    number={1},
    pages={36--52},
    year={2016},
    publisher={IEEE}
    }
- Course concepts: SLAM, Particle Filters, Optimal Control
