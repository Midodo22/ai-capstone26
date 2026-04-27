import os
import re
import glob
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0
DEPTH_SCALE = 1000.0 #

def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Convert inputs to numpy arrays
    rgb   = rgb_image.astype(np.float64) / 255.0
    depth = depth_image.astype(np.float64)

    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    depth = depth / 255.0 * 10.0
    
    # 3. Create a coordinate grid for (u, v) pixels
    h, w = depth.shape
    u_coords = np.arange(w)   # column indices
    v_coords = np.arange(h)   # row indices
    u, v = np.meshgrid(u_coords, v_coords)

    # TODO: Implement unprojection logic here
    z = -depth
    x = (u - CX) * depth / FX
    y = -(v - CY) * depth / FY

    # filter points
    valid = (np.isfinite(z) & (depth > 0.15)& (depth < 4.5))
    points_3d = np.stack((x, y, z), axis=-1)[valid]
    colors_norm = rgb[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # TODO: Estimate normals for pcd_down (required for Point-to-Plane ICP)
    # pcd_down.estimate_normals(...)
    radius_normal = voxel_size * 2.0
    if len(pcd_down.points) > 0:
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        pcd_down.orient_normals_consistent_tangent_plane(30)
    
    # Compute FPFH features for Global Registration [cite: 30]
    radius_feature = voxel_size * 5.0
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%) 
    Implement your own version of Point-to-Plane ICP.
    """
    T_global = initial_transform.copy()
    
    # TODO: Implement the ICP loop:
    # 1. Find nearest neighbors using target_tree.search_knn_vector_3d
    # 2. Build the linear system (AtA)x = Atb
    # 3. Solve for pose update and update T_global
    MAX_ITER = 50
    TOLERANCE = 1e-6
    NEIGHBOR_DIST = 0.5

    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        result = o3d.pipelines.registration.RegistrationResult()
        result.transformation = T_global
        return result

    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    target_points = np.asarray(target_pcd.points)
    target_normals = np.asarray(target_pcd.normals)
    source_points = np.asarray(source_pcd.points)

    for iteration in range(MAX_ITER):
        src_transformed = (T_global[:3, :3] @ source_points.T).T + T_global[:3, 3]
        AtA = np.zeros((6, 6), dtype=np.float64)
        Atb = np.zeros(6, dtype=np.float64)
        num_valid = 0

        for pt in src_transformed:
            k, idx, dist2 = target_tree.search_knn_vector_3d(pt, 1)
            if k == 0 or dist2[0] > NEIGHBOR_DIST ** 2:
                continue

            tgt_pt = target_points[idx[0]]
            normal = target_normals[idx[0]]
            if not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-8:
                continue
            
            # build point to point linear constraint
            row = np.concatenate((np.cross(pt, normal), normal))
            residual = np.dot(normal, tgt_pt - pt)
            
            # accumulate normal equations
            AtA += np.outer(row, row)
            Atb += row * residual
            num_valid += 1

        # stop if fewer than 10 matches
        if num_valid < 10:
            break

        # solve 6D pose update with tiny diagonal regularizer
        try:
            delta = np.linalg.solve(AtA + 1e-8 * np.eye(6), Atb)
        except np.linalg.LinAlgError:
            break

        rot_vec = delta[:3]
        trans_vec = delta[3:]
        rot_norm = np.linalg.norm(rot_vec)
        if rot_norm < 1e-12:
            dR = np.eye(3)
        else:
            dR = R.from_rotvec(rot_vec).as_matrix()

        # form incremental transform dT, multiply to global pose
        dT = np.eye(4)
        dT[:3, :3] = dR
        dT[:3, 3] = trans_vec
        T_global = dT @ T_global

        if np.linalg.norm(delta) < TOLERANCE:
            break
    
    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result

def local_icp_algorithm(source_down, target_down, trans_init, threshold):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    # TODO: Use o3d.pipelines.registration.registration_icp
    # Estimation method should be TransformationEstimationPointToPlane()
    coarse_result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold * 2.0,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40),
    )
    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        threshold,
        coarse_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    return result

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    # extract camera positions
    pred_positions = np.array([pose[:3, 3] for pose in predicted_cam_poses])  # (N, 3)
    
    if isinstance(gt_poses, np.ndarray) and len(gt_poses) > 0:
        gt_positions = gt_poses[:, :3, 3]  # (N, 3)
    elif isinstance(gt_poses, list) and len(gt_poses) > 0:
        gt_positions = np.array([pose[:3, 3] for pose in gt_poses])  # (N, 3)
    else:
        print("Warning: No ground truth poses found, skipping GT trajectory.")
        gt_positions = None
    
    # 1. Create LineSet for estimated trajectory (Red)
    pred_lines = [[i, i+1] for i in range(len(pred_positions) - 1)]
    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_positions)
    pred_lineset.lines  = o3d.utility.Vector2iVector(pred_lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in pred_lines])
    
    # 2. Create LineSet for ground truth trajectory (Black)
    if gt_positions is not None:
        gt_lines = [[i, i+1] for i in range(len(gt_positions) - 1)]
        gt_lineset = o3d.geometry.LineSet()
        gt_lineset.points = o3d.utility.Vector3dVector(gt_positions)
        gt_lineset.lines  = o3d.utility.Vector2iVector(gt_lines)
        gt_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0] for _ in gt_lines]) # Black
    
    # TODO: Calculate Mean L2 Distance between predicted_cam_poses and gt_poses [cite: 38]
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    mean_l2_error = np.nan
    if gt_positions is not None:
        min_len = min(len(pred_positions), len(gt_positions))
        if min_len > 1:
            pred_eval = pred_positions[:min_len]
            gt_eval = gt_positions[:min_len]

            gt_center = np.mean(gt_eval, axis=0)
            pred_center = np.mean(pred_eval, axis=0)
            gt_zero = gt_eval - gt_center
            pred_zero = pred_eval - pred_center
            H = gt_zero.T @ pred_zero
            U, _, Vt = np.linalg.svd(H)
            R_align = Vt.T @ U.T
            if np.linalg.det(R_align) < 0:
                Vt[-1, :] *= -1
                R_align = Vt.T @ U.T
            t_align = pred_center - R_align @ gt_center

            gt_positions = (R_align @ gt_positions.T).T + t_align
            gt_lineset.points = o3d.utility.Vector3dVector(gt_positions)

            gt_eval = gt_positions[:min_len]
            l2_dist = np.linalg.norm(pred_eval - gt_eval, axis=1)
            mean_l2_error = float(np.mean(l2_dist))
            print(f"Mean L2 distance: {mean_l2_error:.6f} meters")
        elif min_len == 1:
            offset = pred_positions[0] - gt_positions[0]
            gt_positions = gt_positions + offset
            gt_lineset.points = o3d.utility.Vector3dVector(gt_positions)
            mean_l2_error = float(np.linalg.norm(pred_positions[0] - gt_positions[0]))
            print(f"Mean L2 distance: {mean_l2_error:.6f} meters")
        else:
            print("Warning: Empty trajectory, skipping L2 evaluation.")
    else:
        print("Warning: Ground truth trajectory unavailable, skipping L2 evaluation.")
    
    # 3. Visualization
    geometries = []
    if len(reconstructed_pcd.points) > 0:
        geometries.append(reconstructed_pcd)
        
    if len(pred_lines) > 0:
        geometries.append(pred_lineset)
    
    if gt_positions is not None and len(gt_lines) > 0:
        geometries.append(gt_lineset)
    
    if len(geometries) > 0:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Floor {args.floor} Reconstruction"
        )
    else:
        print("Warning: Nothing to visualize. Please check that RGB, depth, and GT_pose data were collected.")
    
    return mean_l2_error

def reconstruct(args):
    voxel_size = 0.25 
    rgb_dir = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    camera_poses = [np.eye(4)]
    accumulated_pcd = o3d.geometry.PointCloud()

    # Reconstruction Loop [cite: 29-30]
    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}...")
        # TODO: Implement the full pipeline:
        if i == 1:
            def _frame_key(path):
                match = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
                return int(match.group(1)) if match else path

            rgb_files[:] = sorted(rgb_files, key=_frame_key)
            depth_files[:] = sorted(depth_files, key=_frame_key)

        # Load images
        rgb_prev  = np.asarray(o3d.io.read_image(rgb_files[i-1]))   # (H, W, 3) uint8
        dep_prev  = np.asarray(o3d.io.read_image(depth_files[i-1])) # (H, W) or (H, W, 3)
        rgb_curr  = np.asarray(o3d.io.read_image(rgb_files[i]))
        dep_curr  = np.asarray(o3d.io.read_image(depth_files[i]))
        
        # If depth loaded as (H, W, 3), take just one channel
        if dep_prev.ndim == 3:
            dep_prev = dep_prev[:, :, 0]
        if dep_curr.ndim == 3:
            dep_curr = dep_curr[:, :, 0]

        # 1. Convert RGB-D to PointCloud (Task 1)
        pcd_prev = depth_image_to_point_cloud(rgb_prev, dep_prev)
        pcd_curr = depth_image_to_point_cloud(rgb_curr, dep_curr)
        if i == 1 and len(pcd_prev.points) > 0:
            pcd_prev_world = deepcopy(pcd_prev)
            pcd_prev_world.transform(camera_poses[0])
            accumulated_pcd += pcd_prev_world

        # 2. Preprocess (Voxel/FPFH/Normals)
        reg_voxel = 0.10
        src_down, src_fpfh = preprocess_point_cloud(pcd_curr, reg_voxel)
        tgt_down, tgt_fpfh = preprocess_point_cloud(pcd_prev, reg_voxel)

        coarse_voxel = reg_voxel * 2.0
        src_global, src_global_fpfh = preprocess_point_cloud(pcd_curr, coarse_voxel)
        tgt_global, tgt_global_fpfh = preprocess_point_cloud(pcd_prev, coarse_voxel)
        
        # 3. Execute Global Registration (RANSAC)
        distance_threshold = coarse_voxel * 1.5
        trans_init = np.eye(4) # identity as initial guess
        
        if len(camera_poses) >= 2:
            trans_init = np.linalg.inv(camera_poses[-2]) @ camera_poses[-1]
        
        if len(src_global.points) >= 50 and len(tgt_global.points) >= 50:
            ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src_global, tgt_global,
                src_global_fpfh, tgt_global_fpfh,
                mutual_filter=False,
                max_correspondence_distance=distance_threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=4,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(20000, 500)
            )
            
            if (
                ransac_result.fitness > 0.20
                and ransac_result.inlier_rmse < 0.30
                and np.linalg.norm(ransac_result.transformation[:3, 3]) < 0.8
            ):
                trans_init = ransac_result.transformation
        else:
            ransac_result = o3d.pipelines.registration.RegistrationResult()
            ransac_result.transformation = np.eye(4)
            ransac_result.fitness = 0.0
        
        # 4. Execute Local Registration (ICP - Task 2)
        icp_threshold = reg_voxel * 1.5
        if args.version == 'open3d':
            icp_result = local_icp_algorithm(src_down, tgt_down, trans_init, icp_threshold)
        else:
            icp_result = my_local_icp_algorithm(src_down, tgt_down, trans_init)


        # 5. Update camera_poses and accumulate points
        T_relative = icp_result.transformation
        max_step = max(0.35, 2.0 * np.linalg.norm(trans_init[:3, 3]) + 0.15)
        if args.version == 'open3d':
            if (
                icp_result.fitness < 0.25
                or icp_result.inlier_rmse > 0.30
                or np.linalg.norm(T_relative[:3, 3]) > max_step
            ):
                T_relative = trans_init
        T_world = camera_poses[-1] @ T_relative
        camera_poses.append(T_world)
        
        # Transform cloud to world frame and accumulate
        pcd_curr_world = deepcopy(pcd_curr)
        pcd_curr_world.transform(T_world)
        accumulated_pcd += pcd_curr_world

    # TODO: Post-processing: remove the ceiling [cite: 37]
    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)
    
    if len(points) > 0:
        valid_mask = np.isfinite(points).all(axis=1)
        valid_points = points[valid_mask]
        valid_colors = colors[valid_mask]

        if len(valid_points) > 0:
            up_axis = 1
            if isinstance(gt_poses, np.ndarray) and len(gt_poses) > 0:
                ref_positions = gt_poses[:, :3, 3]
            else:
                ref_positions = np.array([pose[:3, 3] for pose in camera_poses])

            if len(ref_positions) > 0:
                cam_height = np.median(ref_positions[:, up_axis])
            else:
                cam_height = np.median(valid_points[:, up_axis])

            percentile_cap = np.percentile(
                valid_points[:, up_axis],
                98.0 if args.floor == 1 else 75.0
            )
            ceiling_offset = 0.50 if args.floor == 1 else 0.75
            ceiling_threshold = min(cam_height + ceiling_offset, percentile_cap)
            keep_mask = valid_points[:, up_axis] < ceiling_threshold
            filtered_pcd = o3d.geometry.PointCloud()
            filtered_pcd.points = o3d.utility.Vector3dVector(valid_points[keep_mask])
            filtered_pcd.colors = o3d.utility.Vector3dVector(valid_colors[keep_mask])
            accumulated_pcd = filtered_pcd
        else:
            accumulated_pcd.points = o3d.utility.Vector3dVector([])
            accumulated_pcd.colors = o3d.utility.Vector3dVector([])
    else:
        print("Warning: Reconstructed point cloud is empty.")
    
    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)
    
    print(f"Total execution time: {time.time() - start_time:.2f}s") # 
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
