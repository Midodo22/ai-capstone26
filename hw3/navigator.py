import math
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

SCENE_PATH = "../hw0/replica_v1/apartment_0/habitat/mesh_semantic.ply"
SENSOR_HEIGHT = 1.5
SENSOR_WIDTH = 512
SENSOR_HEIGHT_PX = 512
SENSOR_PITCH = 0.0
MOVE_AMOUNT = 0.1
TURN_AMOUNT = 1.0
INITIAL_HEADING = 0.0

# Default action names
MOVE_FORWARD = "move_forward"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"

# =============================
# Image Formatting
# =============================
def _transform_rgb_bgr(image: np.ndarray) -> np.ndarray:
    """Convert RGB to BGR for OpenCV display."""
    return image[:, :, [2, 1, 0]]

def _transform_depth(image: np.ndarray) -> np.ndarray:
    """Normalize and convert depth to a displayable uint8 image."""
    return (image / 10.0 * 255).astype(np.uint8)

def _transform_semantic(semantic_obs: np.ndarray) -> np.ndarray:
    """Convert raw semantic map to a colorized image."""
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    return cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)


def _yaw_to_quaternion(yaw: float) -> List[float]:
    """Return a Habitat-compatible yaw quaternion in (x, y, z, w) order."""
    half_yaw = yaw / 2.0
    return [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)]


def _direction_to_yaw(dx: float, dz: float) -> float:
    """
    Convert a world-space direction into Habitat yaw.
    Habitat agents face along the negative Z axis when yaw == 0.
    """
    return math.atan2(-dx, -dz)


def _compress_waypoints(path_world: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Merge densely-sampled planner pixels into coarser waypoints that are at least
    one forward action apart, while preserving the final goal.
    """
    if len(path_world) <= 2:
        return path_world

    compressed = [path_world[0]]
    anchor = path_world[0]

    for waypoint in path_world[1:-1]:
        dx = waypoint[0] - anchor[0]
        dz = waypoint[1] - anchor[1]
        if math.hypot(dx, dz) >= MOVE_AMOUNT:
            compressed.append(waypoint)
            anchor = waypoint

    if compressed[-1] != path_world[-1]:
        compressed.append(path_world[-1])

    return compressed

# =============================
# Simulator Core
# =============================
def init_sim(scene_path: str = SCENE_PATH, start_x: float = 0.9, start_z: float = 4.6):
    """Initialize the Habitat simulator environment and set the agent's start state."""
    sim_settings = {
        "scene": scene_path,
        "default_agent": 0,
        "sensor_height": SENSOR_HEIGHT,
        "width": SENSOR_WIDTH,
        "height": SENSOR_HEIGHT_PX,
        "sensor_pitch": SENSOR_PITCH,
    }

    # Global Config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = sim_settings["scene"]

    # Agent Config
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    specs = []

    # Define sensors
    for uuid, stype in [
        ("color_sensor", habitat_sim.SensorType.COLOR),
        ("depth_sensor", habitat_sim.SensorType.DEPTH),
        ("semantic_sensor", habitat_sim.SensorType.SEMANTIC),
    ]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = stype
        spec.resolution = [sim_settings["height"], sim_settings["width"]]
        spec.position = [0.0, sim_settings["sensor_height"], 0.0]
        spec.orientation = [sim_settings["sensor_pitch"], 0.0, 0.0]
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        specs.append(spec)

    agent_cfg.sensor_specifications = specs

    # Define action space
    agent_cfg.action_space = {
        MOVE_FORWARD: habitat_sim.agent.ActionSpec(
            MOVE_FORWARD, habitat_sim.agent.ActuationSpec(amount=MOVE_AMOUNT)
        ),
        TURN_LEFT: habitat_sim.agent.ActionSpec(
            TURN_LEFT, habitat_sim.agent.ActuationSpec(amount=TURN_AMOUNT)
        ),
        TURN_RIGHT: habitat_sim.agent.ActionSpec(
            TURN_RIGHT, habitat_sim.agent.ActuationSpec(amount=TURN_AMOUNT)
        ),
    }

    # Initialize Simulator
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Initialize Agent at starting coordinates
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([start_x, 0.0, start_z])  # World translation
    agent_state.rotation = _yaw_to_quaternion(INITIAL_HEADING)
    agent.set_state(agent_state)

    print("Habitat simulator initialized successfully.")
    return sim, agent, list(agent_cfg.action_space.keys())


def navigate_and_see(sim, agent, action: str, goal_index: int = None):
    """
    Step the simulator by one action, display the sensor observations in OpenCV,
    and visually highlight the physical destination.
    """
    if action not in [MOVE_FORWARD, TURN_LEFT, TURN_RIGHT]:
        return

    obs = sim.step(action)
    rgb = _transform_rgb_bgr(obs["color_sensor"])
    depth = _transform_depth(obs["depth_sensor"])
    semantic_labels = obs["semantic_sensor"]

    # Overlay goal label if provided
    if goal_index is not None:
        mask = semantic_labels == np.uint32(goal_index)
        if np.any(mask):
            overlay = rgb.copy()
            overlay[mask] = np.array([0, 0, 255], dtype=overlay.dtype)
            rgb = cv2.addWeighted(overlay, 0.3, rgb, 0.7, 0)

    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth)
    cv2.imshow("Semantic", _transform_semantic(semantic_labels))
    cv2.waitKey(1)

    return obs


def execute_waypoint_path(path_world: List[Tuple[float, float]], sim, agent, goal_idx: int = None):
    """
    Convert a sequence of world 3D waypoints into simulator actuation actions
    (turning and moving forward).
    """
    path_world = _compress_waypoints(path_world)
    heading = INITIAL_HEADING

    for i in range(1, len(path_world)):
        x0, z0 = path_world[i - 1]
        x1, z1 = path_world[i]
        dx, dz = x1 - x0, z1 - z0
        segment_length = math.hypot(dx, dz)

        if segment_length < 1e-6:
            continue

        target_angle = _direction_to_yaw(dx, dz)
        dtheta = (target_angle - heading + math.pi) % (2 * math.pi) - math.pi

        # 1. Turn to align the agent towards the target angle
        turn_steps = int(round(abs(math.degrees(dtheta)) / TURN_AMOUNT))
        if turn_steps > 0:
            action = TURN_LEFT if dtheta > 0 else TURN_RIGHT
            for _ in range(turn_steps):
                navigate_and_see(sim, agent, action, goal_idx)

        # 2. Step forward physically toward the waypoint
        steps_forward = max(1, int(round(segment_length / MOVE_AMOUNT)))
        for _ in range(steps_forward):
            navigate_and_see(sim, agent, MOVE_FORWARD, goal_idx)

        # Update current heading tracker
        heading = target_angle

    print("Agent path execution completed.")
