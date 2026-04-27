import random
import sys
from typing import List, Tuple

from map_processor import load_and_filter_map, select_start, get_goal_pixels
import rrt_utils as rrt

POINT_CLOUD_DATA = "semantic_3d_pointcloud/point.npy"
COLOR_DATA = "semantic_3d_pointcloud/color0255.npy"

# Sample semantic color and index dictionaries for a few object categories. 
# Check hw0/replica_v1/apartment_0/habitat/info_semantic.json and 
# hw3/color_coding_semantic_segmentation_classes.xlsx for the full list of 
# categories and their corresponding colors and indices.
SEMANTIC_DICTS = {
    "colors": {
        "rack": [[0, 255, 133]],
        "cooktop": [[7, 255, 224]],
        "sofa": [[10, 0, 255]],
        "cushion": [[255, 9, 92]],
        "stair": [[173, 255, 0]]
    },
    "indices": {
        "rack": 8,
        "cooktop": 280,
        "sofa": 196,
        "cushion": 431,
        "stair": 192
    },
}


def pick_goal(map_img) -> Tuple[str, Tuple[int, int]]:
    prompt = "Enter semantic destination (ex: 'rack', 'cooktop', 'sofa'): "
    goal_prompt = input(prompt).strip().lower()
    if goal_prompt not in SEMANTIC_DICTS["colors"]:
        print(f"Goal '{goal_prompt}' is not valid.")
        sys.exit(1)

    goal_pixels = get_goal_pixels(map_img, SEMANTIC_DICTS["colors"], goal_prompt)
    goal = random.choice(goal_pixels)
    return goal_prompt, goal


def run_in_sim(start_world: Tuple[float, float], world_path: List[Tuple[float, float]], goal_prompt: str):
    from navigator import init_sim, execute_waypoint_path

    start_x, start_z = start_world
    print(f"Spawning Agent at world position: ({start_x:.3f}, {start_z:.3f})")

    sim, agent, _ = init_sim(start_x=start_x, start_z=start_z)
    execute_waypoint_path(world_path, sim, agent, SEMANTIC_DICTS["indices"][goal_prompt])


def raster_to_world(raster, x_min, z_min, img_height, resolution) -> Tuple[float, float]:
    col, row = raster
    # Map each raster cell to its center in Habitat world coordinates.
    wx = ((col + 0.5) * resolution) + x_min
    wz = ((img_height - row - 0.5) * resolution) + z_min
    return (wx, wz)


def main():
    """Entry point."""

    print("=== Step 1: Processing the 3D Map ===")
    (raster_map, occupancy_map, x_min, z_min, resolution, img_height) = load_and_filter_map(POINT_CLOUD_DATA, COLOR_DATA)


    print("=== Step 2: Selecting Agent Start and Goal Positions ===")
    start = select_start(raster_map)
    goal_prompt, goal = pick_goal(raster_map)
    print(f"Start raster coordinates: {start}")
    print(f"Goal raster coordinates: {goal}")

    print("=== Step 3: Executing Path Planning (RRT) ===")
    path = rrt.plan_path(start, goal, occupancy_map)
    if not path:
        print("Planner could not find a path.")
        sys.exit(1)
    print("Path found.")


    print("=== Step 4: Visualizing the Planned Path ===")
    explored_edges = rrt.get_last_rrt_edges()
    explored_nodes = rrt.get_last_rrt_nodes()
    rrt.visualize_path(
        raster_map,
        path,
        start,
        goal,
        explored_edges=explored_edges,
        explored_nodes=explored_nodes,
    )

    return

    print("=== Step 5: Translating Path to Habitat Simulator ===")
    world_path = [
        raster_to_world(pixel, x_min, z_min, img_height, resolution)
        for pixel in path
    ]

    print("World-space waypoints:")
    for idx, waypoint in enumerate(world_path):
        print(f"  {idx}: ({waypoint[0]:.3f}, {waypoint[1]:.3f})")

    try:
        run_in_sim(world_path[0], world_path, goal_prompt)
    except ModuleNotFoundError as exc:
        print(f"Skipping Habitat navigation because a dependency is missing: {exc}")


if __name__ == "__main__":
    main()
