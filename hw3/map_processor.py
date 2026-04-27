import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

SCALE_FACTOR = 10000.0 / 255.0
CEILING_COLOR = np.array([8, 255, 214])
FLOOR_COLOR = np.array([255, 194, 7])
ROBOT_CLEARANCE_HEIGHT = 1.0
ROBOT_RADIUS_METER = 0.1
WALKABLE_SUPPORT_MAX_HEIGHT = 0.1
OBSTACLE_MIN_HEIGHT = 0.1
MIN_OBSTACLE_COMPONENT_PIXELS = 4
MIN_FLOOR_COMPONENT_PIXELS = 12
DISPLAY_COLOR_WEIGHT_THRESHOLD = 0.05


def splat_point_weights(
    rows_float: np.ndarray,
    cols_float: np.ndarray,
    img_height: int,
    img_width: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize points with bilinear weights onto the four nearest pixels.
    This fixes most visual holes at the projection stage instead of with
    extra post-processing passes.
    """
    row0 = np.floor(rows_float).astype(int)
    col0 = np.floor(cols_float).astype(int)
    drow = rows_float - row0
    dcol = cols_float - col0

    all_rows = []
    all_cols = []
    all_weights = []
    all_indices = []
    point_indices = np.arange(len(rows_float), dtype=int)

    for row_offset, row_weight in ((0, 1.0 - drow), (1, drow)):
        for col_offset, col_weight in ((0, 1.0 - dcol), (1, dcol)):
            rows = row0 + row_offset
            cols = col0 + col_offset
            weights = row_weight * col_weight

            valid = (
                (rows >= 0) & (rows < img_height) &
                (cols >= 0) & (cols < img_width) &
                (weights > 1e-6)
            )
            if np.any(valid):
                all_rows.append(rows[valid])
                all_cols.append(cols[valid])
                all_weights.append(weights[valid].astype(np.float32))
                all_indices.append(point_indices[valid])

    if not all_rows:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
            np.array([], dtype=np.float32),
            np.array([], dtype=int),
        )

    return (
        np.concatenate(all_rows),
        np.concatenate(all_cols),
        np.concatenate(all_weights),
        np.concatenate(all_indices),
    )


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Drop tiny connected components caused by sparse or noisy point samples."""
    if min_area <= 1:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered[labels == label] = 1
    return filtered


def load_and_filter_map(point_path: str, color_path: str):

    points = np.load(point_path)
    colors = np.load(color_path)

    # Convert to real-world meters
    coords = points * SCALE_FACTOR

    # =============== TODO 1-1 ===============
    # Hints: To get a good 2d map, filter ceiling/floor, project to 2D,
    # remove isolated points, inflate obstacles to get occupancy map, etc.
    # IMPORTANT: return map_img as float in value range [0, 1] for visualization downstream.
    # NOTE: in habitat sim, x z plane corresponds to world horizontal plane, and y is vertical.
    
    color_diff = colors.astype(int)
    ceiling_mask = np.all(np.abs(color_diff - CEILING_COLOR) == 0, axis=1)
    floor_mask = np.all(np.abs(color_diff - FLOOR_COLOR) == 0, axis=1)
    floor_height = float(np.median(coords[floor_mask, 1]))
    walkable_height_cutoff = floor_height + WALKABLE_SUPPORT_MAX_HEIGHT
    obstacle_min_height = floor_height + OBSTACLE_MIN_HEIGHT
    obstacle_height_cutoff = floor_height + ROBOT_CLEARANCE_HEIGHT
    walkable_support_mask = (~ceiling_mask) & (coords[:, 1] <= walkable_height_cutoff)
    obstacle_mask = (
        (~ceiling_mask & ~floor_mask)
        & (coords[:, 1] >= obstacle_min_height)
        & (coords[:, 1] <= obstacle_height_cutoff)
    )
    # Rasterize into a top-down grid shared by the visualization and the planner.
    # Hide low walkable-support surfaces such as carpet/rugs in both outputs.
    display_candidate_mask = (~ceiling_mask & ~floor_mask) & (coords[:, 1] >= walkable_height_cutoff)
    display_candidate_coords = coords[display_candidate_mask]
    display_candidate_colors = colors[display_candidate_mask]
    floor_coords = coords[walkable_support_mask]
    obstacle_coords = coords[obstacle_mask]

    xs = display_candidate_coords[:, 0]
    zs = display_candidate_coords[:, 2]
    x_min, x_max = xs.min(), xs.max()
    z_min, z_max = zs.min(), zs.max()

    resolution = 0.05  # meters per pixel
    img_width = int((x_max - x_min) / resolution) + 1
    img_height = int((z_max - z_min) / resolution) + 1

    def project_to_grid(points_3d: np.ndarray):
        if len(points_3d) == 0:
            return (
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )
        cols = (points_3d[:, 0] - x_min) / resolution
        rows = (img_height - 1) - ((points_3d[:, 2] - z_min) / resolution)
        return rows.astype(np.float32), cols.astype(np.float32)

    display_rows, display_cols = project_to_grid(display_candidate_coords)
    floor_rows, floor_cols = project_to_grid(floor_coords)
    obstacle_rows, obstacle_cols = project_to_grid(obstacle_coords)

    # Build a walkable-space map:
    # 0 = free space, 1 = occupied or unknown.
    floor_density = np.zeros((img_height, img_width), dtype=np.float32)
    floor_splat_rows, floor_splat_cols, floor_weights, _ = splat_point_weights(
        floor_rows, floor_cols, img_height, img_width
    )
    if floor_weights.size > 0:
        np.add.at(floor_density, (floor_splat_rows, floor_splat_cols), floor_weights)

    obstacle_density = np.zeros((img_height, img_width), dtype=np.float32)
    obstacle_splat_rows, obstacle_splat_cols, obstacle_weights, _ = splat_point_weights(
        obstacle_rows, obstacle_cols, img_height, img_width
    )
    if obstacle_weights.size > 0:
        np.add.at(obstacle_density, (obstacle_splat_rows, obstacle_splat_cols), obstacle_weights)

    floor_mask_2d = (floor_density > 0.10).astype(np.uint8)
    obstacle_mask_2d = (obstacle_density > 0.08).astype(np.uint8)

    floor_mask_2d = remove_small_components(floor_mask_2d, MIN_FLOOR_COMPONENT_PIXELS)
    obstacle_mask_2d = remove_small_components(obstacle_mask_2d, MIN_OBSTACLE_COMPONENT_PIXELS)

    # Grow walkable support from all low-height surfaces so rugs and doorway
    # thresholds can reopen even when they are not labeled with the exact
    # floor semantic color.
    floor_kernel = np.ones((5, 5), np.uint8)
    obstacle_kernel = np.ones((3, 3), np.uint8)
    floor_region = cv2.morphologyEx(floor_mask_2d, cv2.MORPH_CLOSE, floor_kernel)
    floor_region = cv2.dilate(floor_region, floor_kernel, iterations=1)
    obstacle_region = cv2.morphologyEx(obstacle_mask_2d, cv2.MORPH_CLOSE, obstacle_kernel)
    obstacle_region = cv2.dilate(obstacle_region, obstacle_kernel, iterations=1)

    occupancy_map = np.ones((img_height, img_width), dtype=np.uint8)
    occupancy_map[floor_region > 0] = 0
    occupancy_map[obstacle_region > 0] = 1

    # Inflate obstacles for safer planning while keeping only floor-supported space free.
    if ROBOT_RADIUS_METER > 0:
        inflate_pixels = max(1, int(round(ROBOT_RADIUS_METER / resolution)))
        inflate_kernel = np.ones((2 * inflate_pixels + 1, 2 * inflate_pixels + 1), np.uint8)
        inflated_obstacles = cv2.dilate(obstacle_region, inflate_kernel, iterations=1)
        inflated_occupancy = occupancy_map.copy()
        inflated_occupancy[inflated_obstacles > 0] = 1
    else:
        inflated_occupancy = occupancy_map

    # Apply the same final obstacle decisions back to the original points so
    # filtered obstacle points do not remain blocked only visually.
    display_point_obstacle_flag = obstacle_mask[display_candidate_mask]
    display_rows_int = np.clip(np.floor(display_rows).astype(int), 0, img_height - 1)
    display_cols_int = np.clip(np.floor(display_cols).astype(int), 0, img_width - 1)
    kept_obstacle_points = inflated_occupancy[display_rows_int, display_cols_int] != 0
    display_keep_mask = (~display_point_obstacle_flag) | kept_obstacle_points

    display_coords = display_candidate_coords[display_keep_mask]
    display_colors = display_candidate_colors[display_keep_mask]
    xs = display_coords[:, 0]
    zs = display_coords[:, 2]
    colors_norm = display_colors / 255.0
    display_rows = display_rows[display_keep_mask]
    display_cols = display_cols[display_keep_mask]

    raster_map = np.ones((img_height, img_width, 3), dtype=np.float32)
    color_sum = np.zeros((img_height, img_width, 3), dtype=np.float32)
    color_weight = np.zeros((img_height, img_width), dtype=np.float32)

    raster_rows, raster_cols, raster_weights, raster_indices = splat_point_weights(
        display_rows, display_cols, img_height, img_width
    )
    if raster_weights.size > 0:
        np.add.at(color_sum, (raster_rows, raster_cols), colors_norm[raster_indices] * raster_weights[:, None])
        np.add.at(color_weight, (raster_rows, raster_cols), raster_weights)

    colored_pixels = color_weight > DISPLAY_COLOR_WEIGHT_THRESHOLD
    raster_map[colored_pixels] = (
        color_sum[colored_pixels] / color_weight[colored_pixels, None]
    )

    # Scatter plot rendered to numpy array for select window
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, zs, c=colors_norm, s=1)
    ax.set_title("2D Semantic Map")
    ax.set_xlabel("X (world units)")
    ax.set_ylabel("Z (world units)")
    ax.axis("equal")
    plt.tight_layout()

    # Render figure to a numpy array (RGB)
    fig.canvas.draw()
    x_lim = ax.get_xlim()
    z_lim = ax.get_ylim()
    fig_w, fig_h = fig.canvas.get_width_height()

    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    map_img = buf.astype(np.float32) / 255.0
    plt.close(fig)
    
    return (raster_map, occupancy_map, x_min, z_min, resolution, img_height)

def select_start(map_img: np.ndarray) -> Tuple[int, int]:
    """Display map and return user-clicked start coordinate."""
    start_point = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            start_point.append((x, y))
            print(f"Start selected: ({x}, {y})")

    cv2.namedWindow("Select Start")
    cv2.setMouseCallback("Select Start", mouse_callback)
    print("Click on the map window to select a start location...")

    while True:
        cv2.imshow("Select Start", (map_img * 255).astype(np.uint8))
        key = cv2.waitKey(1) & 0xFF
        if start_point:
            break
        if key == ord("q"):
            raise RuntimeError("No start selected. Exiting.")

    cv2.destroyWindow("Select Start")
    return start_point[0]


def get_goal_pixels(map_img: np.ndarray, semantic_dict: dict, goal_name: str) -> List[Tuple[int, int]]:
    """function to find all pixels corresponding to the goal object based on color matching."""

    if goal_name.lower() not in semantic_dict:
        raise ValueError(f"Unknown semantic object: {goal_name}. Available options: {list(semantic_dict.keys())}")

    goal_colors = semantic_dict[goal_name.lower()]
    goal_pixels: List[Tuple[float, float]] = []

    for gc in goal_colors:
        gc_norm = np.array(gc) / 255.0
        mask_goal = np.all(np.isclose(map_img, gc_norm, atol=10/255.0), axis=2)
        zs, xs = np.where(mask_goal)
        goal_pixels.extend(list(zip(xs, zs)))

    if not goal_pixels:
        raise ValueError(f"No valid pixels found for '{goal_name}'.")

    return goal_pixels
