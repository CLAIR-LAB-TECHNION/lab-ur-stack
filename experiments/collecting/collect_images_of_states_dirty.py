import json
import os
import cv2
import numpy as np
import typer
from lab_ur_stack.motion_planning.motion_planner import MotionPlanner
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1, ur5e_2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
import time
from PIL import Image
from lab_ur_stack.manipulation.utils import ur5e_2_distribute_blocks_in_workspace_uniform, ur5e_2_collect_blocks_from_positions
from lab_ur_stack.utils.workspace_utils import stack_position_r2frame, workspace_x_lims_default, \
    workspace_y_lims_default

sensing_configs = [[-2.04893, -2.50817, -0.00758, -1.96019, 1.51035, 1.0796],
                   [-1.8152335325824183, -2.732894559899801, -0.5337811708450317, -1.1349691313556214,
                    1.0154942274093628, 1.1909868717193604],
                   [-2.29237, -2.50413, -0.76933, -1.37775, 1.53721, -0.74012],
                   [-1.54191, -1.8467, -2.39349, 0.27255, 0.80462, 0.63606],
                   [-1.51508, -2.15687, -1.61078, -0.84643, 0.68978, 1.52553],
                   [-0.13548, -1.67402, -1.56538, -2.86094, 1.10085, 1.63128],
                   [-1.4104, -1.74145, -0.18662, -2.55688, 1.09938, 1.80797]]


app = typer.Typer()
@app.command(
    context_settings={"ignore_unknown_options": True})
def main(n_blocks: int = 3,
         repeat: int = 1):
    camera = RealsenseCamera()

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    r1_controller.speed = 0.75
    r1_controller.acceleration = 0.75
    r2_controller.speed = 1.5
    r2_controller.acceleration = 1.5

    stack_position = gt.point_robot_to_world(ur5e_2["name"], (*stack_position_r2frame, 0.2))

    for i in range(repeat):
        r1_controller.move_home(speed=0.5, acceleration=0.5)

        block_positions = ur5e_2_distribute_blocks_in_workspace_uniform(n_blocks, r2_controller,
                                                                        ws_lim_x=workspace_x_lims_default,
                                                                        ws_lim_y=workspace_y_lims_default,
                                                                        stack_position_ur5e_2_frame=stack_position_r2frame,
                                                                        min_dist=0.06)

        # clear out for images
        r2_controller.plan_and_move_to_xyzrz(stack_position[0], stack_position[1], 0.2, 0)

        # create dir for this run
        datetime = time.strftime("%Y%m%d-%H%M%S")
        run_dir = f"collected_images/{n_blocks}blocks_hires_{datetime}/"
        os.makedirs(run_dir, exist_ok=True)

        # Save block positions
        metadata = {
            "ur5e_1_config": [],
            "ur5e_2_config": [],
            "block_positions": block_positions,
            "images_depth": [],
            "images_rgb": [],
        }

        for idx, c in enumerate(sensing_configs):
            if motion_planner.is_config_feasible("ur5e_1", c) is False:
                print(f"Config {c} is not feasible, probably collides with other robot")
                continue
            r1_controller.plan_and_moveJ(c)
            im, depth = camera.get_frame_bgr()
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            plotable_depth = camera.plotable_depth(depth)

            image_filename_png, image_filename_npy = f"image_{idx}.png", f"image_{idx}.npy"
            depth_filename_png, depth_filename_npy = f"plotable_depth_{idx}.png", f"depth_{idx}.npy"

            Image.fromarray(im).save(os.path.join(run_dir, image_filename_png))
            np.save(os.path.join(run_dir, image_filename_npy), im)
            Image.fromarray(plotable_depth).save(os.path.join(run_dir, depth_filename_png))
            np.save(os.path.join(run_dir, depth_filename_npy), depth)

            # Save metadata
            metadata["ur5e_1_config"].append(c)
            metadata["ur5e_2_config"].append(r2_controller.getActualQ())
            metadata["images_rgb"].append(image_filename_npy)
            metadata["images_depth"].append(depth_filename_npy)

        # Save metadata to JSON
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        r1_controller.move_home(speed=0.5, acceleration=0.5)

        ur5e_2_collect_blocks_from_positions(block_positions, r2_controller,
                                             stack_position_ur5e_2_frame=stack_position_r2frame)


if __name__ == "__main__":
    app()
