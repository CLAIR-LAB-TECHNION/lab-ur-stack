import numpy as np
from robot_inteface.robot_interface import RobotInterfaceWithGripper, home_config
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from utils import logging_util
import time
import logging


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

    def __init__(self, robot_ip, robot_name, motion_palnner: MotionPlanner,
                 geomtry_and_transofms: GeometryAndTransforms, freq=50, gripper_id=0):
        super().__init__(robot_ip, freq, gripper_id)

        logging_util.setup_logging()

        self.robot_name = robot_name
        self.motion_planner = motion_palnner
        self.gt = geomtry_and_transofms

        motion_palnner.visualize()
        time.sleep(0.2)

    @classmethod
    def build_from_robot_name_and_ip(cls, robot_ip, robot_name):
        motion_planner = MotionPlanner()
        geomtry_and_transofms = GeometryAndTransforms(motion_planner)
        return cls(robot_ip, robot_name, motion_planner, geomtry_and_transofms)

    def update_mp_with_current_config(self):
        self.motion_planner.update_robot_config(self.robot_name, self.getActualQ())
        logging.info(f"{self.robot_name} Updated motion planner with current configuration {self.getActualQ()}")

    def find_ik_solution(self, pose, max_tries=10, for_down_movement=True, shoulder_constraint_for_down_movement=0.15):
        """
        if for_down_movement is True, there will be a heuristic check that tha shoulder is not facing down, so when
        movel will be called it won't collide with the table when movingL down.
        """
        # try to find the one that is closest to the current configuration:
        solution = self.getInverseKinematics(pose)
        if solution == []:
            logging.error(f"{self.robot_name} no inverse kinematic solution found at all "
                          f"for pose {pose}")

        def valid_shoulder_angle(q):
            if for_down_movement:
                return -shoulder_constraint_for_down_movement > q[1] > -np.pi + shoulder_constraint_for_down_movement
            else:
                return True

        trial = 1
        while ((self.motion_planner.is_config_feasible(self.robot_name, solution) is False or
               valid_shoulder_angle(solution) is False)
               and trial < max_tries):
            trial += 1
            # try to find another solution, starting from other random configurations:
            qnear = np.random.uniform(-np.pi / 2, np.pi / 2, 6)
            solution = self.getInverseKinematics(pose, qnear=qnear)

        if trial == max_tries:
            logging.error(f"{self.robot_name} Could not find a feasible IK solution after {max_tries} tries")
        elif trial > 1:
            logging.info(f"{self.robot_name} Found IK solution after {trial} tries")
        else:
            logging.info(f"{self.robot_name} Found IK solution in first try")

        return solution

    def plan_and_moveJ(self, q, speed=None, acceleration=None, visualise=True):
        """
        Plan and move to a joint configuration.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        start_config = self.getActualQ()

        logging.info(f"{self.robot_name} planning and movingJ to {q} from {start_config}")

        if visualise:
            self.motion_planner.vis_config(self.robot_name, q, vis_name="goal_config",
                                           rgba=(0, 1, 0, 0.5))
            self.motion_planner.vis_config(self.robot_name, start_config,
                                           vis_name="start_config", rgba=(1, 0, 0, 0.5))

        # plan until the ratio between length and distance is lower than 3, but stop if 4 seconds have passed
        path = self.motion_planner.plan_from_start_to_goal_config(self.robot_name,
                                                                  start_config,
                                                                  q,
                                                                  max_time=4,
                                                                  max_length_to_distance_ratio=2)

        if path is None:
            logging.error(f"{self.robot_name} Could not find a path")
        else:
            logging.info(f"{self.robot_name} Found path with {len(path)} waypoints, moving...")

        if visualise:
            self.motion_planner.vis_path(self.robot_name, path)

        self.move_path(path, speed, acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    def plan_and_move_home(self, speed=None, acceleration=None):
        """
        Plan and move to the home configuration.
        """
        if speed is None:
            speed = self.speed
        if acceleration is None:
            acceleration = self.acceleration

        self.plan_and_moveJ(home_config, speed, acceleration)

    def plan_and_move_to_xyzrz(self, x, y, z, rz, speed=None, acceleration=None, visualise=True,
                               for_down_movement=True):
        """
        if for_down_movement is True, there will be a heuristic check that tha shoulder is not facing down, so when
        movel will be called it won't collide with the table when movingL down.
        Plan and move to a position in the world coordinate system, with gripper
        facing downwards rotated by rz.
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
        self.plan_and_moveJ(goal_config, speed, acceleration, visualise)
        # motion planner is automatically updated after movement

    def pick_up(self, x, y, rz, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} picking up at {x}{y}{rz} with start height {start_height}")
        self.release_grasp()

        # move above pickup location:
        self.plan_and_move_to_xyzrz(x, y, start_height, rz)
        above_pickup_config = self.getActualQ()

        # move down until contact, here we move a little bit slower than drop and sense
        # because the gripper rubber may damage from the object at contact:
        logging.debug(f"{self.robot_name} moving down until contact")
        lin_speed = min(self.linear_speed / 2, 0.05)
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

    def put_down(self, x, y, rz, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} putting down at {x}{y}{rz} with start height {start_height}")
        # move above dropping location:
        self.plan_and_move_to_xyzrz(x, y, start_height, rz, speed=self.speed, acceleration=self.acceleration)
        above_drop_config = self.getActualQ()

        logging.debug(f"{self.robot_name} moving down until contact to put down")
        # move down until contact:
        lin_speed = min(self.linear_speed, 0.05)
        self.moveUntilContact(xd=[0, 0, -lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # release grasp:
        self.release_grasp()
        # back up 10 cm in a straight line :
        self.moveL_relative([0, 0, 0.1], speed=self.linear_speed, acceleration=self.linear_acceleration)
        # move to above dropping location:
        self.moveJ(above_drop_config, speed=self.speed, acceleration=self.acceleration)
        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

    def sense_height(self, x, y, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} sensing height not tilted! at {x}{y} with start height {start_height}")
        self.grasp()

        # move above sensing location:
        self.plan_and_move_to_xyzrz(x, y, start_height, 0, speed=self.speed, acceleration=self.acceleration)
        above_sensing_config = self.getActualQ()

        lin_speed = min(self.linear_speed, 0.1)
        # move down until contact:
        self.moveUntilContact(xd=[0, 0, lin_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
        # measure height:
        height = self.getActualTCPPose()[2]
        # move up:
        self.moveJ(above_sensing_config, speed=self.speed, acceleration=self.acceleration)

        # update the motion planner with the new configuration:
        self.update_mp_with_current_config()

        return height

    def sense_height_tilted(self, x, y, start_height=0.2):
        """
        TODO
        :param x:
        :param y:
        :param rz:
        :param start_height:
        :return:
        """
        logging.info(f"{self.robot_name} sensing height tilted at {x}{y} with start height {start_height}")
        self.grasp()

        # set end effector to be the tip of the finger
        self.setTcp([0.020, 0.012, 0.160, 0, 0, 0])

        logging.debug(f"moving above sensing point with TCP set to tip of the finger")

        # move above point with the tip tilted:
        pose = self.gt.get_tilted_pose_6d_for_sensing(self.robot_name, [x, y, start_height])
        goal_config = self.find_ik_solution(pose, max_tries=50)
        self.plan_and_moveJ(goal_config)

        above_sensing_config = self.getActualQ()

        logging.debug(f"moving down until contact with TCP set to tip of the finger")

        # move down until contact:
        self.moveUntilContact(xd=[0, 0, -self.linear_speed, 0, 0, 0], direction=[0, 0, -1, 0, 0, 0])
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
