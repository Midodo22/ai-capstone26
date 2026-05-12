import argparse, os, json
import numpy as np
import traceback
from scipy.spatial.transform import Rotation as R
from scipy.linalg import pinv

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_ur5_DH_params

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 40
IK_ERROR_THRESH = 0.02

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    """Compute the 3D vector cross product.

    Parameters
    ----------
    a : np.ndarray
        First 3D vector.
    b : np.ndarray
        Second 3D vector.

    Returns
    -------
    np.ndarray
        Cross product ``a x b``.
    """
    return np.cross(a, b)



def _get_initial_q(q_init=None):
    """Validate and normalize an initial 6-DoF joint vector.

    Parameters
    ----------
    q_init : list | tuple | np.ndarray | None
        Initial joint values. Must contain at least 6 numbers.

    Returns
    -------
    np.ndarray
        Joint vector of shape ``(6,)`` in float64.

    Raises
    ------
    ValueError
        If ``q_init`` is missing or has fewer than 6 elements.
    """
    if q_init is not None:
        q_init = np.asarray(q_init, dtype=np.float64).reshape(-1)
        if q_init.size < 6:
            raise ValueError(f"q_init should have at least 6 values, got {q_init.size}")
        return q_init[:6].copy()
    raise ValueError(
        "Cannot infer initial joints. Provide q_init or pass a sequence/articulation object."
    )


def _ik_levenberg_marquardt(dh_params, target_pos, target_rot, base_pos, q_init,
                           joint_limits, max_iters=1000, stop_thresh=0.001):
    """Solve IK using Levenberg-Marquardt method.
    
    Parameters
    ----------
    dh_params : dict
        DH parameters for UR5.
    target_pos : np.ndarray
        Target position (3D).
    target_rot : np.ndarray
        Target rotation matrix (3x3).
    base_pos : np.ndarray
        Base position (3D).
    q_init : np.ndarray
        Initial joint angles (6D).
    joint_limits : np.ndarray
        Joint limits (6x2).
    max_iters : int
        Maximum iterations.
    stop_thresh : float
        Stopping threshold.
    
    Returns
    -------
    np.ndarray
        Solved joint angles (6D).
    """
    tmp_q = q_init.copy()
    step_rate = 0.5
    lambda_param = 0.001  # Initial damping
    lambda_max = 10.0
    lambda_min = 0.00001
    nu = 10.0  # Parameter adjustment factor
    
    for iteration in range(max_iters):
        current_pose, jacobian = your_fk(dh_params, tmp_q, base_pos)
        current_pos = np.asarray(current_pose[:3], dtype=np.float64)
        current_quat = np.asarray(current_pose[3:], dtype=np.float64)
        current_rot = R.from_quat(current_quat).as_matrix()
        
        error_6d, error_norm = _compute_6d_error(target_pos, target_rot, current_pos, current_rot)
        
        if error_norm < stop_thresh:
            break
        
        J = jacobian
        m, n = J.shape
        
        # Levenberg-Marquardt update
        JJt = J @ J.T
        JJt_lm = JJt + lambda_param * np.eye(m)
        
        try:
            J_pinv = J.T @ np.linalg.inv(JJt_lm)
        except np.linalg.LinAlgError:
            J_pinv = pinv(J)
        
        delta_q = J_pinv @ error_6d
        q_new = tmp_q + step_rate * delta_q
        
        # Clip to joint limits
        for i in range(6):
            q_new[i] = np.clip(q_new[i], joint_limits[i, 0], joint_limits[i, 1])
        
        # Evaluate new error
        new_pose, _ = your_fk(dh_params, q_new, base_pos)
        new_pos = np.asarray(new_pose[:3], dtype=np.float64)
        new_quat = np.asarray(new_pose[3:], dtype=np.float64)
        new_rot = R.from_quat(new_quat).as_matrix()
        
        _, new_error_norm = _compute_6d_error(target_pos, target_rot, new_pos, new_rot)
        
        # Adjust damping parameter
        if new_error_norm < error_norm:
            tmp_q = q_new
            lambda_param = max(lambda_param / nu, lambda_min)
        else:
            lambda_param = min(lambda_param * nu, lambda_max)
    
    return list(tmp_q)


def your_ik(new_pose : list or tuple or np.ndarray, 
                base_pos, max_iters : int=1000, stop_thresh : float=.001, q_init=None,
                method : str = "pseudo_inverse"):
    """Solve inverse kinematics using iterative Jacobian pseudo-inverse updates.

    Parameters
    ----------
    new_pose : list | tuple | np.ndarray
        Target end-effector pose in 7D format ``[x, y, z, qx, qy, qz, qw]``.
    base_pos : list | tuple | np.ndarray
        Robot base translation in world frame.
    max_iters : int, default=1000
        Maximum optimization iterations.
    stop_thresh : float, default=0.001
        Stopping threshold on the 6D pose error norm.
    q_init : list | tuple | np.ndarray | None
        Initial joint guess (length >= 6).

    Returns
    -------
    list
        Estimated 6 joint values in radians.

    Homework Hints
    --------------
    Input:
    - Target pose ``new_pose`` and a valid initial guess ``q_init``.
    Output:
    - Joint angles that minimize pose error.

    Suggested implementation logic:
    1. Evaluate current pose and Jacobian via ``your_fk``.
    2. Build 6D error ``[position_error, orientation_error]``.
    3. Compute ``delta_q = pinv(J) @ error``.
    4. Apply step size and clip by joint limits.
    5. Stop when error norm is below threshold.

    Example
    -------
    ```python
    target = [0.4, 0.0, 0.8, 0.0, 0.7071, 0.0, 0.7071]
    q_sol = your_ik(target, base_pos=[-0.2, 0.13, 0.6], q_init=np.zeros(6))
    ```

    Notes
    -----
    Orientation error is computed from relative rotation
    ``R_target @ R_current.T`` converted to axis-angle form.
    """



    joint_limits = np.asarray([
            [-3*np.pi/2, -np.pi/2], # joint1
            [-2.3562, -1],           # joint2
            [-17, 17],              # joint3
            [-17, 17],              # joint4
            [-17, 17],              # joint5
            [-17, 17],              # joint6
        ])

    tmp_q = _get_initial_q(q_init=q_init)
    base_pos = np.asarray(base_pos if base_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        
    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
    # ---          Try to implement `your_ik` without simulator IK APIs           --- #
    # ---          API. (40% for accuracy)                                         --- #
    # --- Note : please modify the code in `your_ik` function.                     --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####

    # TODO: update tmp_q using an iterative optimization loop.
    # tmp_q = ? # may be more than one line
    
    # hint : 
    # 1. You may use `your_fk` function and jacobian matrix to do this
    # 2. Be careful when computing the delta x
    # 3. You may use some hyper parameters (i.e., step rate) in optimization loops

    ###################
    
    dh_params = get_ur5_DH_params()
    target_pose = np.asarray(new_pose, dtype=np.float64)
    
    # Extract target position and orientation
    target_pos = target_pose[:3]
    target_quat = target_pose[3:]  # [qx, qy, qz, qw]
    target_rot = R.from_quat(target_quat).as_matrix()
    
    if method.lower() == "levenberg_marquardt":
        return _ik_levenberg_marquardt(dh_params, target_pos, target_rot, base_pos, tmp_q,
                                       joint_limits, max_iters, stop_thresh)
    
    
    # Hyperparameters for optimization
    step_rate = 0.5  # Learning rate for joint updates
    damping_lambda = 0.01  # Damping factor for pseudo-inverse
    
    # Iterative optimization loop
    for iteration in range(max_iters):
        # 1. Evaluate current pose and Jacobian
        current_pose, jacobian = your_fk(dh_params, tmp_q, base_pos)
        current_pos = np.asarray(current_pose[:3], dtype=np.float64)
        current_quat = np.asarray(current_pose[3:], dtype=np.float64)
        current_rot = R.from_quat(current_quat).as_matrix()
        
        # 2. Compute 6D error
        # Position error (3D)
        pos_error = target_pos - current_pos
        
        # Orientation error (3D axis-angle from relative rotation)
        # R_error = R_target @ R_current.T
        R_error = target_rot @ current_rot.T
        
        # Convert rotation matrix to axis-angle
        try:
            rot_error_obj = R.from_matrix(R_error)
            ori_error = rot_error_obj.as_rotvec()  # axis-angle representation
        except Exception:
            # If rotation matrix is invalid, use small zero error
            ori_error = np.array([0.0, 0.0, 0.0])
        
        # Full 6D error vector
        error_6d = np.concatenate([pos_error, ori_error])
        error_norm = np.linalg.norm(error_6d)
        
        # 3. Check stopping condition
        if error_norm < stop_thresh:
            break
        
        # 4. Compute delta_q using pseudo-inverse with damping
        J = jacobian
        m, n = J.shape  # m=6, n=6
        
        # Compute pseudo-inverse with damping (Levenberg-Marquardt style)
        JJt = J @ J.T
        JJt_damped = JJt + damping_lambda * np.eye(m)
        try:
            J_pinv = J.T @ np.linalg.inv(JJt_damped)
        except np.linalg.LinAlgError:
            # Fall back to simple pseudo-inverse if singular
            J_pinv = pinv(J)
        
        # Compute joint update
        delta_q = J_pinv @ error_6d
        
        # 5. Apply step size and update joints
        tmp_q = tmp_q + step_rate * delta_q
        
        # 6. Clip joints to stay within limits
        for i in range(6):
            tmp_q[i] = np.clip(tmp_q[i], joint_limits[i, 0], joint_limits[i, 1])
    

    return list(tmp_q) # 6 DoF


def score_ik(student_ik_function, headless=False):
    """Run official IK scoring for a student IK function.

    Parameters
    ----------
    student_ik_function : Callable
        Student IK function compatible with
        ``student_ik_function(new_pose, base_pos, q_init=...)``.
    headless : bool, default=False
        Whether Isaac Sim runs without GUI.

    Returns
    -------
    dict
        Score summary including per-file scores and total score.

    Notes
    -----
    The simulator setup and articulation control flow are preserved from the
    original main-loop implementation.
    """
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise ImportError("Isaac Sim python modules are not available in current environment.") from exc

    sim_app = SimulationApp({"headless": bool(headless), "width": 1280, "height": 720})

    try:

        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.api.controllers.articulation_controller import ArticulationController
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.api.world import World
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()


        # Load UR5 into Isaac world using the requested API set.
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac assets root path is None")
        usd_path = assets_root + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        prim_path = "/World/envs/env_0/ur5"

        add_reference_to_stage(usd_path, prim_path)

        robot_view = Articulation(prim_paths_expr=prim_path, name="ur5_view")
        articulation_controller = ArticulationController()

        # Reset after stage edits, then initialize articulation controller.
        world.reset()
        articulation_controller.initialize(robot_view)


        # Match Isaac initial pose to the reference initial joint states.
        reference_init_states = np.asarray([
            -3.141592642791131,
            -1.5707963240621052,
            1.5707963521600738,
            -1.5707963267948966,
            -1.5707963267948966,
            1.06243199169874e-08,
        ], dtype=np.float64)

        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        target_positions = current_positions.copy()
        n_apply = min(target_positions.size, reference_init_states.size)
        target_positions[:n_apply] = reference_init_states[:n_apply]

        # Drive articulation to target initial joint state.
        articulation_controller.apply_action(ArticulationAction(joint_positions=target_positions))
        for _ in range(40):
            world.step(render=not headless)

        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        for _ in range(10):
            world.step(render=not headless)

        testcase_files = [
            'test_case/ik_test_case_easy.json',
            'test_case/ik_test_case_medium.json',
            'test_case/ik_test_case_hard.json',
        ]

        dh_params = get_ur5_DH_params()
        # Keep base frame consistent with the verified Isaac FK test configuration.
        base_pos = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)
        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        if current_positions.size < 6:
            raise RuntimeError(f"UR5 articulation has invalid dof size: {current_positions.size}")
        q_curr = current_positions[:6].copy()
        testcase_file_num = len(testcase_files)
        ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
        ik_error_cnt = [0 for _ in range(testcase_file_num)]

        print("============================ Task 2 : Inverse Kinematic ============================\n")

        for file_id, testcase_file in enumerate(testcase_files):
            try:
                with open(testcase_file, 'r') as f_in:
                    ik_dict = json.load(f_in)
            except Exception:
                traceback.print_exc()
                continue

            test_case_name = os.path.split(testcase_file)[-1]
            poses = ik_dict['next_poses']
            cases_num = len(poses)
            penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)

            ik_errors = []
            for case_id, target_pose in enumerate(poses):
                try:
                    q_sol = student_ik_function(
                        new_pose=target_pose,
                        base_pos=base_pos,
                        q_init=q_curr,
                    )
                    q_curr = np.asarray(q_sol, dtype=np.float64)

                    # Apply IK solution to UR5 articulation in Isaac Sim.
                    current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
                    target_positions = current_positions.copy()
                    target_positions[:6] = q_curr
                    action = ArticulationAction(joint_positions=target_positions)
                    articulation_controller.apply_action(action)

                    # Let articulation move before evaluating end-effector pose.
                    for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                        world.step(render=not headless)

                    solved_pose, _ = your_fk(dh_params, q_curr, base_pos)
                    ik_error = np.linalg.norm(np.asarray(solved_pose) - np.asarray(target_pose), ord=2)
                    ik_errors.append(ik_error)
                    if ik_error > IK_ERROR_THRESH:
                        ik_score[file_id] -= penalty
                        ik_error_cnt[file_id] += 1
                except Exception:
                    traceback.print_exc()
                    world.step(render=not headless)
                    continue


            ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
            ik_errors = np.asarray(ik_errors)
            mean_file_err = float(np.mean(ik_errors)) if ik_errors.size > 0 else float('nan')

            score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                        "- Mean Error : {:0.06f}\n".format(mean_file_err) + \
                        "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                        "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                                ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)
            print(score_msg)

        total_ik_score = 0.0
        for file_id in range(testcase_file_num):
            total_ik_score += ik_score[file_id]

        print("====================================================================================")
        print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score , TASK2_SCORE_MAX))
        print("====================================================================================")

        return {
            "ik_score": ik_score,
            "ik_error_count": ik_error_cnt,
            "total_score": total_ik_score,
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:

        sim_app.close()


def main(args):
    """CLI entry point for IK homework evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line options. ``headless`` is forwarded to the scorer.

    Returns
    -------
    dict
        Score summary from ``score_ik``.
    """
    # Create a wrapper function that uses the specified method
    def your_ik_with_method(new_pose, base_pos, max_iters=1000, stop_thresh=0.001, q_init=None):
        return your_ik(new_pose, base_pos, max_iters=max_iters, stop_thresh=stop_thresh, 
                      q_init=q_init, method=args.method)
    
    return score_ik(your_ik_with_method, headless=bool(args.headless))
    
    # return score_ik(your_ik, headless=bool(args.headless))
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False,
                        help='run Isaac Sim without rendering window')
    parser.add_argument('--method', type=str, default='pseudo_inverse',
                        choices=['pseudo_inverse', 'levenberg_marquardt'],
                        help='IK method to use (default: pseudo_inverse)')
    
    args = parser.parse_args()
    main(args)