import numpy as np
from robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from utils import logging_util
import time
import logging
import chime


def canonical_last_joint_config(config):
    """
    Adjusts the last joint angle of a robotic configuration to a canonical range.

    This function ensures that the last joint angle of the input configuration
    (i.e., the sixth element) is normalized to within the range [-π, π]. It performs
    this adjustment using modular arithmetic until the angle falls within the
    desired range.

    Parameters:
        config (list or numpy array): A sequence representing the joint
        configuration of a robotic system, where the last element specifies
        the angle of the sixth joint.

    Returns:
        list or numpy array: The adjusted configuration with the last joint
        angle normalized to the range [-π, π].
    """
    while config[5] > np.pi:
        config[5] -= 2 * np.pi

    while config[5] < -np.pi:
        config[5] += 2 * np.pi

    return config

def to_valid_limits_config(config):
    """
    Adjusts each element of the input configuration to be within the range of -2*pi to 2*pi.

    Parameters:
    config (list): A list of numeric values representing the input configuration.

    Returns:
    list: A modified list where each value is adjusted to fall within the range of -2*pi to 2*pi.
    """
    for i in range(6):
        while config[i] >= 2 * np.pi:
            config[i] -= 2 * np.pi

        while config[i] <= - 2 * np.pi:
            config[i] += 2 * np.pi

    return config

class ManipulationController(RobotInterfaceWithGripper):
    """
    Extension for the RobotInterfaceWithGripper to higher level manipulation actions and motion planning.
    """
    # those are angular in radians:
    speed = 1.0
    acceleration = 1.0

    # and this is linear, ratio that makes sense:
    @property
    def linear_speed(self):
        return self.speed * 0.1

    @property
    def linear_acceleration(self):
        return self.acceleration * 0.1

    def __init__(self, robot_ip, robot_name, motion_planner: MotionPlanner,
                 geometry_and_transforms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, freq, gripper_id)

        logging_util.setup_logging()

        self.robot_name = robot_name
        self.motion_planner = motion_planner
        self.gt = geometry_and_transforms

        # Add window name to distinguish between different visualizations
        if not MotionPlanner.vis_initialized:
            motion_planner.visualize(window_name="robots_visualization")

        self.setTcp([0, 0, 0.150, 0, 0, 0])

        motion_planner.visualize()
        time.sleep(0.2)

        chime.theme('pokemon')

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geometry_and_transforms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geometry_and_transforms)

    def update_mp_with_current_config(self):
        """
        Updates the motion planner with the robot's current configuration.

        This method retrieves the robot's current configuration and updates the motion planner with the new configuration for the robot. Logs the updated configuration for reference.
        """
        self.motion_planner.update_robot_config(self.robot_name, self.getActualQ())
        logging.info(f"{self.robot_name} Updated motion planner with current configuration {self.getActualQ()}")

    def find_ik_solution(self, pose, max_tries=10, for_down_movement=True, shoulder_constraint_for_down_movement=0.3):
        """
        Find an inverse kinematics (IK) solution for a given pose, with constraints on joint limits and safety.

        Parameters:
        - pose: Target pose for which the IK solution is to be found.
        - max_tries: Maximum number of attempts to find a feasible IK solution.
        - for_down_movement: Boolean flag indicating whether the motion is for downward movement.
        - shoulder_constraint_for_down_movement: Constraint on the shoulder joint range for downward movements.

        Returns:
        - A valid and feasible IK solution that is closest to the current configuration and satisfies constraints, or None if no solution is found after the specified number of attempts.

        Behavior:
        - Attempts to find an IK solution closest to the current configuration.
        - Checks if the configuration is safe based on the specified constraints.
        - Verifies the feasibility of the solution using a motion planner and safety checks.
        - If no feasible solution is found within max_tries, logs an error.
        - Logs information about the number of attempts taken to find the solution.

        Notes:
        - The safety of the configuration is assessed differently based on whether the motion involves downward movement.
        - The final solution is adjusted using canonical and validity functions to ensure it is within proper joint limits.
        """
        # try to find the one that is closest to the current configuration:
        solution = self.getInverseKinematics(pose)
        if solution == []:
            logging.error(f"{self.robot_name} no inverse kinematic solution found at all "
                          f"for pose {pose}")

        def is_safe_config(q):
            if for_down_movement:
                safe_shoulder = -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement
                safe_for_sensing_close = True
                # if 0 > pose[1] > -0.4 and -0.1 < pose[0] < 0.1:  # too close to robot base
                #     print(pose)
                #     safe_for_sensing_close = -3*np.pi/4 < q[0] < -np.pi/2 or np.pi/2 < q[0] < 3*np.pi/4
                return safe_shoulder and safe_for_sensing_close
            else:
                return True

        trial = 1
        while ((self.motion_planner.is_config_feasible(self.robot_name, solution) is False or
               is_safe_config(solution) is False)
               and trial < max_tries):
            trial += 1
            # try to find another solution, starting from other random configurations:
            q_near = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
            solution = self.getInverseKinematics(pose, qnear=q_near)

        solution = canonical_last_joint_config(solution)
        solution = to_valid_limits_config(solution)

        if trial == max_tries:
            logging.error(f"{self.robot_name} Could not find a feasible IK solution after {max_tries} tries")
            return None
        elif trial > 1:
            logging.info(f"{self.robot_name} Found IK solution after {trial} tries")
        else:
            logging.info(f"{self.robot_name} Found IK solution in first try")

        return solution

    def plan_and_moveJ(self, goal_config, speed=None, acceleration=None, visualise=True):
        """
        Plans and executes a joint-space motion for the robot to move from its current configuration to the specified goal configuration.

        Parameters:
        goal_config: The target joint configuration for the robot to reach.
        speed: Optional; The speed at which the robot should execute the motion. If not provided, the default speed is used.
        acceleration: Optional; The acceleration to be applied during the motion. If not provided, the default acceleration is used.
        visualise: Optional; A flag to indicate whether the motion planning and execution process should be visualized. Default is True.

        Returns:
        bool: True if the path was successfully planned and executed, False otherwise.

        Raises:
        None

        Behavior:
        1. Initializes the speed and acceleration with default values if not explicitly provided.
        2. Logs the current action of planning and moving the robot.
        3. If visualization is enabled, visualizes the start and goal configurations.
        4. Plans a path from the current configuration to the goal configuration. The planning terminates either when an acceptable path is found or a maximum time of 8 seconds is reached.
        5. If a valid path is found:
           - Logs the success and number of waypoints in the path.
           - Visualizes the planned path if visualization is enabled.
           - Executes the motion along the planned path with the specified speed and acceleration.
           - Updates the motion planner with the new robot configuration.
           - Returns True.
        6. If no valid path is found:
           - Logs the failure to find a path.
           - Outputs a message indicating the failure.
           - Returns False.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        start_config = self.getActualQ()

        logging.info(f"{self.robot_name} planning and movingJ to {goal_config} from {start_config}")

        if visualise:
            self.motion_planner.vis_config(self.robot_name, goal_config, vis_name="goal_config",
                                           rgba=(0, 1, 0, 0.5))
            self.motion_planner.vis_config(self.robot_name, start_config,
                                           vis_name="start_config", rgba=(1, 0, 0, 0.5))

        # plan until the ratio between length and distance is lower than 2, but stop if 8 seconds have passed
        path = self.motion_planner.plan_from_start_to_goal_config(self.robot_name,
                                                                  start_config,
                                                                  goal_config,
                                                                  max_time=8,
                                                                  max_length_to_distance_ratio=2)

        if path is None:
            logging.error(f"{self.robot_name} Could not find a path")
            print("Could not find a path, not moving.")
            return False
        else:
            logging.info(f"{self.robot_name} Found path with {len(path)} waypoints, moving...")

        if visualise:
            self.motion_planner.vis_path(self.robot_name, path)

        self.move_path(path, speed, acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()
        return True

    def plan_and_move_home(self, speed=None, acceleration=None):
        """
        Plans and executes a motion to move the robot to the predefined home configuration.

        Parameters:
        speed (float, optional): The speed at which the robot should move. If not provided, the default speed set for the object is used.
        acceleration (float, optional): The acceleration to use during the motion. If not provided, the default acceleration set for the object is used.

        The method utilizes the `plan_and_moveJ` function with the predefined home configuration, specified speed, and acceleration to plan and carry out the movement.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        self.plan_and_moveJ(home_config, speed, acceleration)

    def plan_and_move_to_xyzrz(self, x, y, z, rz, speed=None, acceleration=None, visualise=True,
                               for_down_movement=True):
        """
        Plans and moves the robot's gripper to a specified position and orientation in the world coordinate system.

        This method computes a target pose with the gripper facing downwards, rotated by the specified rz angle,
        and moves the robot gripper to the desired position. If the movement requires the robot's shoulder to face
        downwards, the method includes a heuristic check to ensure that the movement does not lead to a collision
        with the table.

        Parameters:
        x: float
            The x-coordinate of the desired position in the world coordinate system.
        y: float
            The y-coordinate of the desired position in the world coordinate system.
        z: float
            The z-coordinate of the desired position in the world coordinate system.
        rz: float
            Rotation angle around the z-axis for the desired orientation in radians.
        speed: Optional[float]
            The speed of the movement. Defaults to the instance's predefined speed if not specified.
        acceleration: Optional[float]
            The acceleration of the movement. Defaults to the instance's predefined acceleration if not specified.
        visualise: bool
            Specifies whether to visualize the planned movement. Default is True.
        for_down_movement: bool
            If True, performs a heuristic check on the shoulder position to prevent collisions with the table
            for downward movements.

        Returns:
        Any
            The result of the plan_and_moveJ function call, which executes the movement.

        Logs:
        Logs the planning and calculated robot frame pose for the given input position and orientation.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        target_pose_robot = self.gt.get_gripper_facing_downwards_6d_pose_robot_frame(self.robot_name,
                                                                                     [x, y, z],
                                                                                     rz)
        logging.info(f"{self.robot_name} planning and moving to xyzrz={x}{y}{z}{rz}. "
                     f"pose in robot frame:{target_pose_robot}")

        shoulder_constraint = 0.15 if z < 0.2 else 0.35
        goal_config = self.find_ik_solution(target_pose_robot, max_tries=50, for_down_movement=for_down_movement,)
        return self.plan_and_moveJ(goal_config, speed, acceleration, visualise)
        # motion planner is automatically updated after movement

    def pick_up(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        Handles the robot's pick-up operation by moving to a specified location,
        grasping the object, and updating the robot's motion planner configuration.

        Parameters:
        x (float): The x-coordinate of the pickup location.
        y (float): The y-coordinate of the pickup location.
        rz (float): The rotation about the z-axis at the pickup location.
        start_height (float, optional): The initial height above the pickup location. Defaults to 0.2.
        replan_from_home_if_failed (bool, optional): Whether to replan from the home position if the pickup operation fails. Defaults to True.

        Procedure:
        - Plans and moves the robot to a position above the specified pickup location.
        - If initial planning and movement fail:
          - Checks `replan_from_home_if_failed`. If True, the robot replans from its home position and retries the movement.
          - If the replanned movement also fails, the function exits with an error.
        - Retrieves and stores the robot's joint configuration above the pickup location.
        - Releases any current grasp to prepare for pickup.
        - Moves the robot down towards the object until contact is detected, using a reduced linear speed to minimize risk of gripper damage.
        - Retracts slightly to avoid surface scratching.
        - Grasps the object using the gripper.
        - Moves back to the stored joint configuration above the pickup location.
        - Updates the motion planner with the robot's current configuration.

        Notes:
        - A weight measurement and success determination step is outlined as a TODO item.
        - Logs key operational details, warnings, and errors during execution.
        """
        logging.info(f"{self.robot_name} picking up at {x}{y}{rz} with start height {start_height}")

        # move above pickup location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz)
            if not res:
                chime.error()
                return

        above_pickup_config = self.getActualQ()
        self.release_grasp()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])

        # retract one more centimeter to avoid gripper scratching the surface:
        self.moveL_relative([0, 0, 0.01],
                            speed=0.1,
                            acceleration=0.1)
        logging.debug(f"{self.robot_name} grasping and picking up")
        # close gripper:
        self.grasp()
        # move up:
        self.moveJ(above_pickup_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

        # TODO measure weight and return if successful or not

    def put_down(self, x, y, rz, start_height=0.2, replan_from_home_if_failed=True):
        """
        Moves the robot to place an object at a specified location.

        Parameters:
        x : float
            X-coordinate of the desired drop location.
        y : float
            Y-coordinate of the desired drop location.
        rz : float
            Rotation about the z-axis at the desired drop location.
        start_height : float, optional
            Height above the drop location from where the robot starts moving. Default is 0.2.
        replan_from_home_if_failed : bool, optional
            Indicates whether to replan the path from the home position if the initial path planning fails. Default is True.

        Behavior:
        - Attempts to move the robot above the specified drop location at the given height and orientation using the plan_and_move_to_xyzrz method.
        - If the motion fails and replan_from_home_if_failed is True, the method replans the movement from the home position.
        - Moves the robot downwards until contact is detected to place the object securely.
        - Releases the grasp on the object.
        - Moves the robot back up by 10 cm in a straight line.
        - Returns the robot to the position above the drop location (pre-drop configuration).
        - Updates the motion planner with the new configuration after placing the object.
        - Logs relevant information about the actions performed, including warnings if initial planning fails.

        Notes:
        - The function uses the robot's configured speed and acceleration parameters for movement.
        - If the movement cannot be planned even after replanning from home, the method exits with an error indication.
        - This function assumes the presence of a properly calibrated motion planner and appropriate implementations for methods such as plan_and_move_to_xyzrz, moveUntilContact, and release_grasp.
        """
        logging.info(f"{self.robot_name} putting down at {x}{y}{rz} with start height {start_height}")
        # move above dropping location:
        res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
            if not res:
                chime.error()
                return

        above_drop_config = self.getActualQ()

        logging.debug(f"{self.robot_name} moving down until contact to put down")
        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # release grasp:
        self.release_grasp()
        # back up 10 cm in a straight line :
        self.moveL_relative([0, 0, 0.1], speed=self.linear_speed, acceleration=self.linear_acceleration)
        # move to above dropping location:
        self.moveJ(above_drop_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    # def sense_height(self, x, y, start_height=0.2):
    #     """
    #     TODO
    #     :param x:
    #     :param y:
    #     :param start_height:
    #     :return:
    #     """
    #     logging.info(f"{self.robot_name} sensing height not tilted! at {x}{y} with start height {start_height}")
    #     self.grasp()
    #
    #     # move above sensing location:
    #     self.plan_and_move_to_xyzrz(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
    #     above_sensing_config = self.getActualQ()
    #
    #     lin_speed = min(self.linear_speed, 0.1)
    #     # move down until contact:
    #     self.moveUntilContact(xd=[0, 0, lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
    #     # measure height:
    #     height = self.getActualTCPPose()[2]
    #     # move up:
    #     self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)
    #
    #     # update the motion planner with the new configuration:
    #     self.update_mp_with_current_config()
    #
    #     return height

    def sense_height_tilted(self, x, y, start_height=0.15, replan_from_home_if_failed=True):
        """
        Senses the height at a specified (x, y) coordinate while the end effector is tilted.

        Parameters:
        x: Coordinate in the x-axis where the height is to be sensed.
        y: Coordinate in the y-axis where the height is to be sensed.
        start_height: The initial height from which to begin the sensing operation. Defaults to 0.15.
        replan_from_home_if_failed: Boolean flag to determine if the robot should attempt to replan from its home position if the initial plan fails. Defaults to True.

        Process Details:
        - Logs the robot's name and sensing operation configuration.
        - Sets the tool center point (TCP) to the tip of the finger for accurate sensing.
        - Calculates and moves to an initial tilted pose above the sensing point.
        - Plans and executes a joint configuration to reach the calculated pose.
        - Handles path-finding failure by optionally returning to home position and retrying.
        - Moves downwards with a contact-sensitive motion until an obstacle is encountered.
        - Measures the height by recording the current TCP pose.
        - Restores the robot's position to above the sensing point.
        - Resets the TCP to its original configuration and updates the motion planner.

        Returns:
        Height sensed at the specified coordinate. Returns -1 if the operation fails after retries.

        Logs:
        Contains debugging and warning logs for tracing the sensing process and any encountered issues.
        """
        logging.info(f"{self.robot_name} sensing height tilted at {x}{y} with start height {start_height}")
        self.grasp()

        # set end effector to be the tip of the finger
        self.setTcp([0.02, 0.012, 0.15, 0, 0, 0])

        logging.debug(f"moving above sensing point with TCP set to tip of the finger")

        # move above point with the tip tilted:
        pose = self.gt.get_tilted_pose_6d_for_sensing(self.robot_name, [x, y, start_height])
        goal_config = self.find_ik_solution(pose, max_tries=50)
        res = self.plan_and_moveJ(goal_config)

        if not res:
            if not replan_from_home_if_failed:
                chime.error()
                return -1

            logging.warning(f"{self.robot_name} replanning from home, probably couldn't find path"
                            f" from current position")
            self.plan_and_move_home()
            res = self.plan_and_moveJ(goal_config)
            if not res:
                chime.error()
                return -1

        above_sensing_config = self.getActualQ()

        logging.debug(f"moving down until contact with TCP set to tip of the finger")

        # move down until contact:
        lin_speed = min(self.linear_speed, 0.04)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        pose = self.getActualTCPPose()
        height = pose[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        # set back tcp:
        self.setTcp([0, 0, 0.150, 0, 0, 0])
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

        logging.debug(f"height measured: {height}, TCP pose at contact: {pose}")

        return height
