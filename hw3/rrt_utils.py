import math
import random
from collections import deque
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np


LAST_RRT_EDGES: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
LAST_RRT_NODES: List[Tuple[float, float]] = []


class Nodes:
    def __init__(self, x: float, y: float, parent: Optional["Nodes"] = None):
        self.x = int(x)
        self.y = int(y)
        self.parent = parent


def collision(x1, y1, x2, y2, occupancy_map):
    """Return True if a segment hits an obstacle or exits the map."""
    height, width = occupancy_map.shape

    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    dx = x2 - x1
    dy = y2 - y1
    steps = max(int(math.hypot(dx, dy)), 1)

    for t in np.linspace(0.0, 1.0, steps + 1):
        x = int(round(x1 + t * dx))
        y = int(round(y1 + t * dy))

        if x < 0 or x >= width or y < 0 or y >= height:
            return True
        if occupancy_map[y, x] != 0:
            return True

    return False


def check_collision(x1, y1, x2, y2, goal, occupancy_map, step_size):
    """
    Grow from node (x2, y2) toward sample (x1, y1) by one step.
    Returns the new point and two booleans:
    1. directCon: new point can connect directly to goal
    2. nodeCon: new point can connect to the parent node
    """
    dist, theta = dist_and_angle(x2, y2, x1, y1)
    height, width = occupancy_map.shape
    step = min(step_size, dist)
    x = x2 + step * math.cos(theta)
    y = y2 + step * math.sin(theta)

    if x < 0 or x >= width or y < 0 or y >= height:
        return (x2, y2, False, False)

    x = int(round(x))
    y = int(round(y))

    if x < 0 or x >= width or y < 0 or y >= height:
        return (x2, y2, False, False)

    nodeCon = not collision(x2, y2, x, y, occupancy_map)
    directCon = nodeCon and not collision(x, y, goal[0], goal[1], occupancy_map)
    return (x, y, directCon, nodeCon)


def dist_and_angle(x1, y1, x2, y2):
    dist = math.hypot(x1 - x2, y1 - y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    return (dist, angle)


def nearest_node(x, y, node_list: List[Nodes]):
    distances = [dist_and_angle(x, y, node.x, node.y)[0] for node in node_list]
    return int(np.argmin(distances))


def rnd_point(h, w):
    return (random.randint(0, w - 1), random.randint(0, h - 1))


def is_free(point: Tuple[int, int], occupancy_map: np.ndarray) -> bool:
    x, y = point
    height, width = occupancy_map.shape
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return occupancy_map[y, x] == 0


def nearest_free_point(point: Tuple[int, int], occupancy_map: np.ndarray, max_radius: int = 25):
    """Move a blocked start/goal to the nearest free pixel."""
    if is_free(point, occupancy_map):
        return point

    px, py = point
    height, width = occupancy_map.shape
    for radius in range(1, max_radius + 1):
        x_min = max(0, px - radius)
        x_max = min(width - 1, px + radius)
        y_min = max(0, py - radius)
        y_max = min(height - 1, py + radius)

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if max(abs(x - px), abs(y - py)) == radius and occupancy_map[y, x] == 0:
                    return (x, y)
    return None


def connected_component_mask(
    start: Tuple[int, int], occupancy_map: np.ndarray
) -> Optional[np.ndarray]:
    """Return the 4-connected free-space component containing start."""
    start_free = nearest_free_point(start, occupancy_map)
    if start_free is None:
        return None

    width = occupancy_map.shape[1]
    height = occupancy_map.shape[0]
    component = np.zeros((height, width), dtype=bool)
    queue = deque([start_free])
    component[start_free[1], start_free[0]] = True

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if component[ny, nx] or occupancy_map[ny, nx] != 0:
                continue
            component[ny, nx] = True
            queue.append((nx, ny))

    return component


def nearest_reachable_goal(
    start: Tuple[int, int],
    goal_points: List[Tuple[int, int]],
    occupancy_map: np.ndarray,
    max_radius: int = 35,
) -> Optional[Tuple[int, int]]:
    """
    Choose a free goal pixel near the semantic target that is reachable from start.
    """
    component = connected_component_mask(start, occupancy_map)
    if component is None:
        return None

    height, width = occupancy_map.shape
    best_goal = None
    best_score = None

    for gx, gy in goal_points:
        for radius in range(max_radius + 1):
            x_min = max(0, gx - radius)
            x_max = min(width - 1, gx + radius)
            y_min = max(0, gy - radius)
            y_max = min(height - 1, gy + radius)

            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    if max(abs(x - gx), abs(y - gy)) != radius:
                        continue
                    if not component[y, x]:
                        continue

                    score = abs(x - gx) + abs(y - gy)
                    if best_score is None or score < best_score:
                        best_goal = (x, y)
                        best_score = score
            if best_goal is not None and best_score == radius:
                break

    return best_goal


def backtrack_path(node: Nodes, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    path = [(int(goal[0]), int(goal[1]))]
    current = node
    while current is not None:
        path.append((int(current.x), int(current.y)))
        current = current.parent
    path.reverse()
    return path


def densify_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Insert intermediate integer pixels so the path reaches every segment endpoint."""
    if not path:
        return []

    dense_path = [path[0]]
    for start, end in zip(path, path[1:]):
        x1, y1 = start
        x2, y2 = end
        steps = max(int(math.hypot(x2 - x1, y2 - y1)), 1)

        for step in range(1, steps + 1):
            t = step / steps
            x = int(round(x1 + t * (x2 - x1)))
            y = int(round(y1 + t * (y2 - y1)))
            if dense_path[-1] != (x, y):
                dense_path.append((x, y))

    return dense_path


def path_is_collision_free(path: List[Tuple[int, int]], occupancy_map: np.ndarray) -> bool:
    if not path:
        return False
    for i in range(len(path) - 1):
        if collision(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], occupancy_map):
            return False
    return True


def RRT(start, goal, occupancy_map, stepSize=10, max_iter=5000, goal_sample_rate=0.15):
    global LAST_RRT_EDGES, LAST_RRT_NODES
    h, w = occupancy_map.shape
    node_list: List[Nodes] = [Nodes(start[0], start[1])]
    LAST_RRT_EDGES = []
    LAST_RRT_NODES = [start]

    for _ in range(max_iter):
        if random.random() < goal_sample_rate:
            nx, ny = goal
        else:
            nx, ny = rnd_point(h, w)

        nearest_ind = nearest_node(nx, ny, node_list)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y

        tx, ty, directCon, nodeCon = check_collision(
            nx, ny, nearest_x, nearest_y, goal, occupancy_map, stepSize
        )

        if not nodeCon:
            continue

        if (tx, ty) == (nearest_x, nearest_y):
            continue

        if any(node.x == tx and node.y == ty for node in node_list):
            continue

        new_node = Nodes(tx, ty, parent=node_list[nearest_ind])
        node_list.append(new_node)
        LAST_RRT_EDGES.append(((nearest_x, nearest_y), (tx, ty)))
        LAST_RRT_NODES.append((tx, ty))

        if directCon:
            LAST_RRT_EDGES.append(((tx, ty), goal))
            path = densify_path(backtrack_path(new_node, goal))
            if path_is_collision_free(path, occupancy_map):
                return path

    return None


def plan_path(start, goal, occupancy_map):
    global LAST_RRT_EDGES, LAST_RRT_NODES
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    LAST_RRT_EDGES = []
    LAST_RRT_NODES = [start]

    start_free = nearest_free_point(start, occupancy_map)
    goal_free = nearest_free_point(goal, occupancy_map)

    if start_free is None or goal_free is None:
        return None

    if not collision(start_free[0], start_free[1], goal_free[0], goal_free[1], occupancy_map):
        return densify_path([start_free, goal_free])

    step_size = max(6, min(15, min(occupancy_map.shape) // 40))
    for _ in range(3):
        path = RRT(start_free, goal_free, occupancy_map, stepSize=step_size, max_iter=8000)
        if path and path[-1] != goal_free and not collision(
            path[-1][0], path[-1][1], goal_free[0], goal_free[1], occupancy_map
        ):
            path = densify_path(path + [goal_free])
        if path_is_collision_free(path, occupancy_map) and path[-1] == goal_free:
            return path
    return None


def get_last_rrt_edges():
    return LAST_RRT_EDGES.copy()


def get_last_rrt_nodes():
    return LAST_RRT_NODES.copy()


def visualize_path(map_img, path, start, goal, explored_edges=None, explored_nodes=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(map_img)

    if explored_edges:
        for p1, p2 in explored_edges:
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="black",
                linewidth=1,
                alpha=0.35,
                zorder=1,
            )

    if explored_nodes:
        node_xs = [p[0] for p in explored_nodes]
        node_ys = [p[1] for p in explored_nodes]
        ax.scatter(node_xs, node_ys, c="black", s=6, alpha=0.45, zorder=2)

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    ax.plot(xs, ys, color="#b11226", linewidth=2.5, zorder=3)

    ax.scatter(start[0], start[1], c="lime", s=70, label="Start", zorder=4)
    ax.scatter(goal[0], goal[1], c="royalblue", s=70, label="Goal", zorder=4)

    ax.set_title("Planned Path on Semantic Map")
    ax.axis("off")
    ax.legend()
    plt.tight_layout()
    plt.savefig('planned_path.png')
    plt.show()
