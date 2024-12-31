"""
Transformations between different coordinate systems, where the world coordinate system is
the one in the motion planner (aligned with UR5e_1)
"""

from klampt.math import se3, so3
import numpy as np
from numpy import pi
from scipy.spatial.transform import Rotation as R
from camera.configurations_and_params import camera_in_ee
from motion_planning.motion_planner import MotionPlanner
from motion_planning.abstract_motion_planner import AbstractMotionPlanner


class GeometryAndTransforms:
    """
    Class that provides geometric transformations and utility functions for converting between various coordinate frames, such as world, robot, end-effector, and camera frames.

    It also provides methods for generating specific poses for robotics applications based on input configurations and coordinate data.
    """
    def __init__(self, motion_planner: AbstractMotionPlanner, cam_in_ee=camera_in_ee):
        self.motion_planner = motion_planner
        self.robot_name_mapping = motion_planner.robot_name_mapping
        self.camera_in_ee = cam_in_ee

    @classmethod
    def from_motion_planner(cls, motion_planner):
        """
        Creates an instance of the class using a motion planner object.

        Parameters:
        motion_planner: An instance of a motion planner that will be used to create the new instance.

        Returns:
        An instance of the class initialized with the provided motion planner.
        """
        return cls(motion_planner)

    @classmethod
    def build(cls):
        """
        Builds an instance of the class with an initialized MotionPlanner object.

        This method acts as a factory that creates a new instance of the class, initializing
        it with a MotionPlanner object.

        Returns:
            An instance of the class initialized with a MotionPlanner object.
        """
        mp = MotionPlanner()
        return cls(mp)

    def point_world_to_robot(self, robot_name, point_world):
        """
        Transforms a point in the world frame to the robot's local frame.

        Parameters:
        robot_name (str): The name of the robot.
        point_world (list or tuple): A 3D point specified in the world frame.

        Returns:
        list: The transformed 3D point in the robot's local frame.
        """
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = se3.inv(robot.link(0).getTransform())
        return se3.apply(world_to_robot, point_world)

    def point_robot_to_world(self, robot_name, point_robot):
        """
        Transforms a point from the robot's coordinate frame to the world coordinate frame.

        This method takes a point specified in the robot's local coordinate frame and converts it
        into the corresponding point in the world coordinate frame by applying the transformation
        matrix associated with the robot's base link.

        Parameters:
        robot_name (str): The name of the robot for which the transformation is to be performed.
        point_robot (tuple or list): The 3D point in the robot's local coordinate frame to be transformed.

        Returns:
        tuple: The transformed 3D point in the world coordinate frame.
        """
        robot = self.robot_name_mapping[robot_name]
        world_to_robot = robot.link(0).getTransform()
        return se3.apply(world_to_robot, point_robot)

    def world_to_robot_ee_transform(self, robot_name, config):
        """
        Computes the transformation from the world frame to the robot's end-effector frame.

        This function calculates the inverse of the transformation matrix that maps the robot's end-effector frame to the world frame, effectively providing the transformation from the world frame to the robot's end-effector frame.

        Parameters:
        robot_name: Name of the robot as a string.
        config: Configuration data or parameters for the robot's transformation computation.

        Returns:
        A transformation matrix representing the world-to-end-effector frame transform.
        """
        ree_2_w = self.robot_ee_to_world_transform(robot_name, config)
        return se3.inv(ree_2_w)

    def robot_ee_to_world_transform(self, robot_name, config):
        """
        Calculates the end-effector to world transformation matrix for a given robot.

        Parameters:
        robot_name (str): The name of the robot for which the transformation is to be calculated.
        config (list): The configuration of the robot, which typically includes joint states or parameters.

        Returns:
        ee_transform: The forward kinematics result, representing the transformation matrix of the robot's end-effector relative to the world frame.
        """
        return self.motion_planner.get_forward_kinematics(robot_name, config)

    def camera_to_ee_transform(self,):
        """
        Calculates the transformation matrix from the camera frame to the end-effector (EE) frame.

        The camera frame is assumed to have the orientation where the Z-axis is forward, the X-axis is to the right,
        and the Y-axis is downward. This matches the EE frame's orientation. Hence, only a translation operation is
        required to compute the transformation.

        Returns:
            SE3: The transformation matrix from the camera frame to the EE frame.
        """
        # we assume camera is z forward, x right, y down (like in the image). this is already the ee frame orientation,
        # so we just need to translate it
        return se3.from_translation(np.array(self.camera_in_ee))

    def ee_to_camera_transform(self, ):
        """
        Transforms the end-effector frame of reference to the camera frame of reference.

        This transformation is based on the given assumption that the camera orientation is aligned such that the z-axis points forward, the x-axis points to the right, and the y-axis points downward. Since this orientation matches the end-effector frame, the transformation only involves translation. The translation vector used for this transformation is the negative of `camera_in_ee`.

        Returns:
            The SE3 transformation matrix representing the translation from the end-effector frame to the camera frame.
        """
        # we assume camera is z forward, x right, y down (like in the image). this is already the ee frame orientation,
        # so we just need to translate it
        return se3.from_translation(-np.array(self.camera_in_ee))

    def world_to_camera_transform(self, robot_name, config):
        """
        Computes the transformation matrix from the world frame to the camera frame.

        Parameters:
        robot_name: The name of the robot for which the transformation is being calculated.
        config: Configuration data required for determining transformations.

        Returns:
        The transformation matrix that defines the relationship from the world frame to the camera frame.
        """
        transform_w_to_ee = self.world_to_robot_ee_transform(robot_name, config)
        transform_ee_to_camera = self.ee_to_camera_transform()
        return se3.mul(transform_ee_to_camera, transform_w_to_ee)

    def camera_to_world_transform(self, robot_name, config):
        """
        Computes the transformation matrix from the camera's coordinate system to the world coordinate system.

        Parameters:
        robot_name: str
            The name of the robot for which the transformation is being calculated.
        config: dict
            A dictionary containing the relevant configuration settings for the robot.

        Returns:
        numpy.ndarray or similar
            A transformation matrix that maps coordinates from the camera's reference frame to the world reference frame.
        """
        transform_camera_to_ee = self.camera_to_ee_transform()
        transform_ee_to_w = self.robot_ee_to_world_transform(robot_name, config)
        return se3.mul(transform_ee_to_w, transform_camera_to_ee)

    def point_world_to_camera(self, point_world, robot_name, config):
        """
        Transforms a point from world coordinates to camera coordinates.

        Args:
        point_world: The point in world coordinates to be transformed.
        robot_name: The name of the robot for which the transformation is being computed.
        config: Configuration settings used to determine the world-to-camera transformation.

        Returns:
        The point in camera coordinates after applying the transformation.
        """
        transform_w_to_camera = self.world_to_camera_transform(robot_name, config)
        return se3.apply(transform_w_to_camera, point_world)

    def point_camera_to_world(self, point_camera, robot_name, config):
        """
        Converts a point from the camera's coordinate system to the world coordinate system.

        Parameters:
        point_camera: The 3D point in the camera's coordinate frame.
        robot_name: The name of the robot for which the transformation is calculated.
        config: Configuration or parameter data required to determine the transformation.

        Returns:
        The 3D point transformed into the world coordinate frame.
        """
        transform_camera_to_w = self.camera_to_world_transform(robot_name, config)
        return se3.apply(transform_camera_to_w, point_camera)

    def get_gripper_facing_downwards_6d_pose_robot_frame(self, robot_name, point_world, rz):
        """
        Computes the 6D pose in the robot frame for a gripper that is facing downwards.

        Parameters:
        robot_name: The name of the robot for which the 6D pose is to be calculated.
        point_world: The 3D point in the world coordinate system for which the pose is required.
        rz: The rotation angle about the z-axis in radians.

        Returns:
        A numpy array containing the concatenated 3D position in the robot frame and the rotation vector (as a 3-element array) representing the gripper's orientation expressed in 6D pose format.
        """
        point_robot = self.point_world_to_robot(robot_name, point_world)

        rotation_down = R.from_euler('xyz', [np.pi, 0, 0])
        rotation_z = R.from_euler('z', rz)
        combined_rotation = rotation_z * rotation_down
        # after spending half day on it, It turns out that UR works with rotvec :(
        r = combined_rotation.as_rotvec(degrees=False)

        return np.concatenate([point_robot, r])

    def get_tilted_pose_6d_for_sensing(self, robot_name, point_world):
        """
        Computes a tilted 6D pose for a robotic end-effector tailored for sensing applications.

        The function determines the desired pose in 6D space based on a specified world point relative to the robot.
        It calculates a rotation around the z-axis, factoring in proximity to the robot base to avoid self-collisions
        or enable further reach. This method assumes specific configurations for 'ur5e_2' settings.

        Parameters:
        robot_name: Name of the robot for which the pose is being calculated.
        point_world: A 3D point specified in the world coordinate system.

        Returns:
        A numpy array representing the 6D pose of the end-effector, consisting of the 3D position of the target point
        in the robot's frame of reference and the rotation vector.
        """
        point_robot = self.point_world_to_robot(robot_name, point_world)

        # rotation of ee around z depends on proximity to the robot base. Too close to the
        # base requires 0 so the robot won't collide with itself, but with 180 it can reach further
        # these numbers are only correct for ur5e_2 in our setting:
        rz = 0 if -0.4 < point_robot[1] < 0.4 else pi

        rotation_euler = R.from_euler('xyz', [-0.8 * pi, 0.15*pi, rz])

        r = rotation_euler.as_rotvec(degrees=False)

        return np.concatenate([point_robot, r])

    def rotvec_to_so3(self, rotvec):
        return so3.from_rotation_vector(rotvec)

    def se3_to_4x4(self, se3_transform):
        return np.array(se3.homogeneous(se3_transform))

    def mat4x4_to_se3(self, mat4x4):
        return se3.from_homogeneous(mat4x4)
