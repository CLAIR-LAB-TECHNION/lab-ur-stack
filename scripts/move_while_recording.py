import time
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
import numpy as np
from matplotlib import pyplot as plt
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCameraWithRecording


def main():
    camera = RealsenseCameraWithRecording()
    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed, r1_controller.acceleration = 0.75, 0.75
    r2_controller.speed, r2_controller.acceleration = 1.0, 1.0

    # Start recording
    camera.start_recording("./test_video", max_depth=5, fps=30)

    try:
        r1_controller.plan_and_move_home()
        color_frame, depth_frame = camera.get_frame_rgb()
        plt.imshow(color_frame)
        plt.show()

        r1_controller.plan_and_moveJ([0.014, -1.644, 0.293, -0.979, -1.022, -0.656])
        color_frame, depth_frame = camera.get_frame_rgb()
        plt.imshow(color_frame)
        plt.show()

        r2_controller.plan_and_move_home()
        color_frame, depth_frame = camera.get_frame_rgb()
        plt.imshow(color_frame)
        plt.show()

        r2_controller.plan_and_move_to_xyzrz(-0.7, -0.7, 0.3, 0)
        color_frame, depth_frame = camera.get_frame_rgb()
        plt.imshow(color_frame)
        plt.show()

        r1_controller.plan_and_move_home()

    finally:
        camera.stop_recording()

if __name__ == "__main__":
    main()