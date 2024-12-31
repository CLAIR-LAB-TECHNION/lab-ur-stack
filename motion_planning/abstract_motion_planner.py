import math
import sys
import time
from abc import abstractmethod
from unittest import skipIf
from frozendict import frozendict
import numpy as np
from numpy import pi
from klampt import WorldModel, Geometry3D, RobotModel
from klampt.model.geometry import box
from klampt import vis
from klampt.plan.cspace import MotionPlan
from klampt.plan import robotplanning
from klampt.model import ik
from klampt.math import se3, so3
from klampt.model import collide
import os

from sympy.codegen.ast import continue_
from trimesh.path.packing import visualize



class AbstractMotionPlanner:
    default_attachments = frozendict(ur5e_1=["camera", "gripper"], ur5e_2=["gripper"])
    default_settings = frozendict({
        "type": "rrt*",
        "bidirectional": False,
        "connectionThreshold": 30.0,
        "perturbationRadius": 1.,
        # "suboptimalityFactor": 1.01,  # only for rrt* and prm*.
        # Don't use suboptimalityFactor as it's unclear how that parameter works...
        # seems like it's ignored even in rrt*
        # "shortcut": True, # only for rrt
    })
    # Class-level attribute to track initialization
    vis_initialized = False

    def __init__(self, eps=2e-2, attachments=default_attachments, settings=default_settings, ee_offset=0.15):
        """
        parameters:
        eps: epsilon gap for collision checking along the line in configuration space. Too high value may lead to
            collision, too low value may lead to slow planning. Default value is 1e-2.
        """
        self.eps = eps
        self.objects = {}
        self.world = WorldModel()
        world_path = self._get_klampt_world_path()
        self.world.readFile(world_path)

        self.ee_offset = ee_offset
        self.ur5e_1 = self.world.robot("ur5e_1")
        self.ur5e_2 = self.world.robot("ur5e_2")
        self.robot_name_mapping = {"ur5e_1": self.ur5e_1, "ur5e_2": self.ur5e_2}
        for robot in self.robot_name_mapping.values():
            self._set_ee_offset(robot)
        self._add_attachments(self.ur5e_1, attachments["ur5e_1"])
        self._add_attachments(self.ur5e_2, attachments["ur5e_2"])

        self.world_collider = collide.WorldCollider(self.world)

        self.settings = frozendict(self.default_settings)
        self.held_objects = {robot_name: None for robot_name in self.robot_name_mapping}  # Track held objects

    def is_pyqt5_available(self):
        try:
            import PyQt5
            return True
        except ImportError:
            return False

    def visualize(self, backend=None, window_name=None):
        """
        Initializes and visualizes the motion planning environment using the specified rendering backend and optional window name.

        Parameters:
        - backend: Optional. The rendering backend to use for visualization. Defaults to "GLUT" on Linux systems or "PyQt5"/"GLUT" on others depending on availability.
        - window_name: Optional. Name of the visualization window. If provided, it will be set as the created window's name.

        Behavior:
        - The function checks if visualization has already been initialized and exits if true.
        - Determines the visualization backend based on the system platform and availability.
        - Initializes the visualization environment with the specified or default backend.
        - Adds the motion planning world to the visualization environment.
        - Sets colors for specific elements in the visualization environment.
        - Configures the camera's position, rotation, and distance for optimal viewing.
        - Displays the visualization and pauses briefly to ensure rendering is shown.
        - Marks the visualization as initialized to prevent re-initialization in subsequent calls.
        """
        if AbstractMotionPlanner.vis_initialized:
            return

        if backend is None:
            if sys.platform.startswith('linux'):
                backend = "GLUT"
            else:
                backend = "PyQt5" if self.is_pyqt5_available() else "GLUT"

        vis.init(backend)
        if window_name:
            vis.createWindow(window_name)

        vis.add("world", self.world)
        vis.setColor(('world', 'ur5e_1'), 0.8, 0.8, 0.8)
        vis.setColor(('world', 'ur5e_2'), 0.8, 0.8, 0.8)

        # set camera position:
        viewport = vis.getViewport()
        viewport.camera.tgt = [0, -0.6, 0.5]
        viewport.camera.rot = [0, -0.75, 1]
        viewport.camera.dist = 5

        vis.show()
        time.sleep(0.2)

        AbstractMotionPlanner.vis_initialized = True

    def vis_config(self, robot_name, config_, vis_name="robot_config", rgba=(0, 0, 1, 0.5)):
        """
        Creates and visualizes a robot configuration in Klampt's visualization module.

        Parameters:
        robot_name (str): The name of the robot for which the configuration will be visualized.
        config_ (list): The robot configuration to be visualized. If the configuration length is 6, it will be converted using the config6d_to_klampt method.
        vis_name (str, optional): The name identifier for the visualization element. Defaults to "robot_config".
        rgba (tuple, optional): The color and transparency for the visualization in RGBA format. Defaults to (0, 0, 1, 0.5).

        Behavior:
        - If the configuration length is 6, it converts the configuration to Klampt's format using config6d_to_klampt.
        - Modifies the configuration into a list of length 1 to work around a visualization limitation.
        - Adds the visual representation of the configuration using Klampt's vis module.
        - Sets the color and attributes for the visualization element.
        """
        config = config_.copy()
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        config = [config]  # There's a bug in visualize config so we just visualize a path of length 1

        vis.add(vis_name, config)
        vis.setColor(vis_name, *rgba)
        vis.setAttribute(vis_name, "robot", robot_name)

    def vis_path(self, robot_name, path_):
        """
        Visualizes a robot's planned trajectory path in the Klamp't visualization environment.

        Parameters:
        robot_name (str): The name of the robot whose path is to be visualized.
        path_ (list): A list of configurations representing the trajectory path, where each configuration can either be in 6D or another form suitable for conversion.

        Behavior:
        1. If the configurations in the input path have six elements, they are converted using the 'config6d_to_klampt' function.
        2. Retrieves the robot instance associated with the given robot_name from the `robot_name_mapping`.
        3. Sets the robot's configuration to the initial configuration of the path.
        4. Displays the trajectory in the Klamp't visualization environment with appropriate attributes:
           - Adds the path as a visual element.
           - Sets the path color to white with partial transparency.
           - Associates the path with the specified robot name.
        """
        path = path_.copy()
        if len(path[0]) == 6:
            path = [self.config6d_to_klampt(q) for q in path]

        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(path[0])
        robot_id = robot.id

        # trajectory = RobotTrajectory(robot, range(len(path)), path)
        vis.add("path", path)
        vis.setColor("path", 1, 1, 1, 0.5)
        vis.setAttribute("path", "robot", robot_name)

    def show_point_vis(self, point, name="point"):
        """
        Displays a 3D point in the visualizer with a given name and specific color properties.

        Parameters:
        point: The 3D point to be displayed in the visualization.
        name: Optional; The identifier for the point in the visualizer. Default is "point".

        The point is added to the visualizer and its color is set to red with partial transparency.
        """
        vis.add(name, point)
        vis.setColor(name, 1, 0, 0, 0.5)

    def show_ee_poses_vis(self):
        """
        Displays the end-effector poses of all robots in the visualization.

        For each robot in the robot name mapping, retrieves the end-effector (EE) transform using the "ee_link" and adds it to the visualization with a unique identifier.

        Parameters:
        None

        Returns:
        None
        """
        for robot in self.robot_name_mapping.values():
            ee_transform = robot.link("ee_link").getTransform()
            vis.add(f"ee_pose_{robot.getName()}", ee_transform)

    def update_robot_config(self, robot_name, config):
        if len(config) == 6:
            config = self.config6d_to_klampt(config)
        robot = self.robot_name_mapping[robot_name]
        robot.setConfig(config)

    def plan_from_start_to_goal_config(self, robot_name: str, start_config, goal_config, max_time=15,
                                       max_length_to_distance_ratio=10):
        """
        Plans a path for a robot from a specified start configuration to a goal configuration.

        Parameters:
        robot_name: The name of the robot for which the planning needs to be executed.
        start_config: The starting configuration of the robot. It could either be a 6D representation or a Klampt configuration.
        goal_config: The target configuration of the robot. It could either be a 6D representation or a Klampt configuration.
        max_time: The maximum allowable time for the planning process, default is 15 seconds.
        max_length_to_distance_ratio: The maximum allowable ratio of path length to the direct distance, default is 10.

        Returns:
        The planned path from start to goal configuration in 6D representation.

        Notes:
        If start_config or goal_config is in 6D, it will be automatically converted to Klampt configuration for planning.
        """
        if len(start_config) == 6 and len(goal_config) == 6:
            start_config = self.config6d_to_klampt(start_config)
            goal_config = self.config6d_to_klampt(goal_config)

        robot = self.robot_name_mapping[robot_name]
        path = self._plan_from_start_to_goal_config_klampt(robot, start_config, goal_config,
                                                           max_time, max_length_to_distance_ratio)

        return self.path_klampt_to_config6d(path)

    def _plan_from_start_to_goal_config_klampt(self, robot, start_config, goal_config, max_time=15,
                                               max_length_to_distance_ratio=10):
        """
        Plans a path for a robot to move from a start configuration to a goal configuration using the Klampt library.

        Parameters:
        robot : The robot for which the path is being planned. The robot's configuration will be updated to the start configuration before planning.
        start_config : The initial configuration of the robot.
        goal_config : The target configuration the robot should reach.
        max_time : Maximum time allowed for the path planning process (default is 15 seconds).
        max_length_to_distance_ratio : Maximum ratio of the path length to the direct distance between the start and goal configurations (default is 10).

        Returns:
        list : A list of configurations representing the planned path from the start configuration to the goal configuration. If a direct path is possible, the list will contain only the goal configuration.

        Notes:
        - A direct path check is performed before planning. If the direct path is feasible, planning is skipped, and the method returns the goal configuration as the path.
        - If a direct path is not possible, the method performs path planning based on the specified settings and parameters.
        """
        robot.setConfig(start_config)

        planner = robotplanning.plan_to_config(self.world, robot, goal_config,
                                               # ignore_collisions=[('keep_out_from_ur3_zone', 'table2')],
                                               # extraConstraints=
                                               **self.settings)
        planner.space.eps = self.eps

        # before planning, check if a direct path is possible, then no need to plan
        if self._is_direct_path_possible(planner, start_config, goal_config):
            return [goal_config]

        return self._plan(planner, max_time, max_length_to_distance_ratio=max_length_to_distance_ratio)

    def _plan(self, planner: MotionPlan, max_time=15, steps_per_iter=1000, max_length_to_distance_ratio=10):
        """
        find path given a prepared planner, with endpoints already set
        @param planner: MotionPlan object, endpoints already set
        @param max_time: maximum planning time
        @param steps_per_iter: steps per iteration
        @param max_length_to_distance_ratio: maximum length of the pass to distance between start and goal. If there is
            still time, the planner will continue to plan until this ratio is reached. This is to avoid long paths
            where the robot just moves around because non-optimal paths are still possible.
        """
        start_time = time.time()
        path = None
        print("planning motion...", end="")
        while (path is None or self.compute_path_length_to_distance_ratio(path) > max_length_to_distance_ratio) \
                and time.time() - start_time < max_time:
            print(".", end="")
            planner.planMore(steps_per_iter)
            path = planner.getPath()
        print("")
        print("planning took ", time.time() - start_time, " seconds.")
        if path is None:
            print("no path found")
        return path

    def plan_multiple_robots(self):
        # implement if\when necessary.
        # robotplanning.plan_to_config supports list of robots and goal configs
        raise NotImplementedError

    @staticmethod
    def config6d_to_klampt(config):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        config_klampt = [0] * 8
        config_klampt[1:7] = config
        return config_klampt

    @staticmethod
    def klampt_to_config6d(config_klampt):
        """
        There are 8 links in our rob for klampt, some are stationary, actual joints are 1:7
        """
        if config_klampt is None:
            return None
        return config_klampt[1:7]

    def path_klampt_to_config6d(self, path_klampt):
        """
        convert a path in klampt 8d configuration space to 6d configuration space
        """
        if path_klampt is None:
            return None
        path = []
        for q in path_klampt:
            path.append(self.klampt_to_config6d(q))
        return path

    def compute_path_length(self, path):
        """
        compute the length of the path
        """
        if path is None:
            return np.inf
        length = 0
        for i in range(len(path) - 1):
            length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
        return length

    def compute_path_length_to_distance_ratio(self, path):
        """ compute the ratio of path length to the distance between start and goal """
        if path is None:
            return np.inf
        start = np.array(path[0])
        goal = np.array(path[-1])
        distance = np.linalg.norm(start - goal)
        length = self.compute_path_length(path)
        return length / distance

    @abstractmethod
    def _add_attachments(self, robot, attachments):
        pass

    def _is_direct_path_possible(self, planner, start_config_, goal_config_):
        # EmbeddedRobotCspace only works with the active joints:
        start_config = self.klampt_to_config6d(start_config_)
        goal_config = self.klampt_to_config6d(goal_config_)
        return planner.space.isVisible(start_config, goal_config)

    def is_config_feasible(self, robot_name, config):
        """
        check if the config is feasible (not within collision)
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        if len(config) == 0:
            return False

        robot = self.robot_name_mapping[robot_name]
        current_config = robot.getConfig()
        robot.setConfig(config_klampt)

        # we have to get all collisions since there is no method for robot-robot collisions-+--
        all_collisions = list(self.world_collider.collisions())

        robot.setConfig(current_config)  # return to original motion planner state

        # all collisions is a list of pairs of colliding geometries. Filter only those that contains a name that
        # ends with "link" and belongs to the robot, and it's not the base link that always collides with the table.
        for g1, g2 in all_collisions:
            if g1.getName().endswith("link") and g1.getName() != "base_link" and g1.robot().getName() == robot_name:
                return False
            if g2.getName().endswith("link") and g2.getName() != "base_link" and g2.robot().getName() == robot_name:
                return False

        return True

    def get_forward_kinematics(self, robot_name, config):
        """
        Calculates the forward kinematics for a specified robot and configuration.

        This method computes the end-effector's pose (position and orientation) in
        the world frame, given a robot's configuration. The configuration can be in
        6D space or in the full configuration space of the robot.

        Parameters:
        robot_name: str
            The name of the robot for which the kinematics is to be calculated.
        config: list or array-like
            The configuration of the robot. If the configuration's length is 6, it's
            assumed to be in a simplified 6D space and will be transformed to the
            full configuration space. Otherwise, it is assumed to already be in the
            full configuration space.

        Returns:
        list
            A 4x4 transformation matrix representing the position and orientation of
            the end-effector in the world frame.
        """
        if len(config) == 6:
            config_klampt = self.config6d_to_klampt(config)
        else:
            config_klampt = config.copy()

        robot = self.robot_name_mapping[robot_name]

        previous_config = robot.getConfig()
        robot.setConfig(config_klampt)
        link = robot.link("ee_link")
        ee_transform = link.getTransform()
        robot.setConfig(previous_config)

        return ee_transform

    def _set_ee_offset(self, robot):
        ee_transform = robot.link("ee_link").getParentTransform()
        ee_transform = se3.mul(ee_transform, (so3.identity(), (0, 0, self.ee_offset)))
        robot.link("ee_link").setParentTransform(*ee_transform)
        # reset the robot config to update:
        robot.setConfig(robot.getConfig())

    def ee_transform_from_point(self, point, orientation=None):
        """
        Returns the end-effector transformation matrix given a point and orientation.

        Parameters:
        robot_name (str): The name of the robot.
        point (list or tuple): The 3D coordinates of the end-effector.
        orientation (list or tuple, optional): The orientation of the end-effector in 3D space.

        Returns:
        list: A 4x4 transformation matrix representing the position and orientation of the end-effector.
        """
        if orientation is None:
            orientation = so3.identity()
        return [orientation, point]

    def ik_solve(self, robot_name, ee_transform, start_config=None):
        """
        Solves the inverse kinematics (IK) problem for a given robot and end-effector transform.

        Parameters:
        robot_name (str): Name of the robot for which the IK solution is to be computed.
        ee_transform (list or tuple): Transform representing the desired position and orientation of the end-effector in 3D space.
        start_config (list or tuple, optional): Optional starting configuration for the IK solver. It should be a list of 6 elements if provided.

        Returns:
        list: The computed configuration in 6D format that satisfies the IK constraints for the given end-effector transform.

        Raises:
        KeyError: If the provided robot_name is not found in the robot_name_mapping.
        ValueError: If start_config is not None and does not contain exactly 6 elements.
        """

        if start_config is not None and len(start_config) == 6:
            start_config = self.config6d_to_klampt(start_config)

        robot = self.robot_name_mapping[robot_name]
        return self.klampt_to_config6d(self._ik_solve_klampt(robot, ee_transform, start_config))

    def _ik_solve_klampt(self, robot, ee_transform, start_config=None):

        curr_config = robot.getConfig()
        if start_config is not None:
            robot.setConfig(start_config)

        ik_objective = ik.objective(robot.link("ee_link"), R=ee_transform[0], t=ee_transform[1])
        res = ik.solve(ik_objective, tol=2e-5, iters=10000)
        if not res:
            # print("ik not solved")
            robot.setConfig(curr_config)
            return None

        res_config = robot.getConfig()

        robot.setConfig(curr_config)

        return res_config


    def add_object_to_world(self, name, item):
        """
        Add a new object to the world.
        :param name: Name of the object.
        :param item: Dictionary containing the following keys:
            - geometry_file: Path to the object's geometry file.
            - coordinates: [x, y, z] coordinates.
            - angle: Rotation matrix (so3).
            - color: Dictionary with 'name' and 'rgb' keys for the object's color (default is white).
            - scale: Scaling factor of the object (default is 1,1,1).
        """

        obj = self.world.makeRigidObject(name)
        geom = obj.geometry()
        if not geom.loadFile(item["geometry_file"]):
            raise ValueError(f"Failed to load geometry file: {item['geometry_file']}")

        # Set the transformation (rotation + position)
        if len(item["angle"]) != 9:
            item["angle"] = so3.rotation(item["angle"], math.pi / 2)
        transform = (item["angle"], item["coordinates"])
        geom.setCurrentTransform(*transform)
        if type(item["scale"]) is float or int:
            geom.scale(item["scale"])
        else:
            geom.scale(*item["scale"])

        # Set the transformation for the rigid object
        obj.setTransform(*transform)

        # Set the object's color
        obj.appearance().setColor(*item["color"]["rgb"])

        # Save the object in the dictionary
        self.objects[name] = obj

        return obj


    def get_object(self, name):
        """
        Retrieve an object by name from the dictionary.
        :param name: Name of the object.
        :return: The object if found, otherwise None.
        """
        obj = self.objects.get(name)
        if obj is None:
            print(f"Object '{name}' not found.")
        return obj

    def remove_object(self, name, vis_state=False):
        """
        Remove an object from the world and the dictionary.
        :param name: Name of the object to be removed.
        :param vis_state: Boolean to visualize the workspace after removing the object.
        """
        if vis.shown():
            vis_state = True
            vis.show(False)
            time.sleep(0.3)
        self._remove_object(name)
        if vis_state:
            self.visualize(window_name="workspace")

    def _remove_object(self, name):
        """
        Remove an object from the world and the dictionary.
        :param name: Name of the object to be removed.
        """
        obj = self.objects.pop(name, None)  # Remove from the dictionary
        if obj is None:
            print(f"Object '{name}' not found. Cannot remove.")
        else:
            self.world.remove(obj)
            print(f"Object '{name}' removed from the dictionary and world.")


    @abstractmethod
    def _get_klampt_world_path(self):
        pass
