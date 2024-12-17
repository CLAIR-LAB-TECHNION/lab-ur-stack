from utils.workspace_utils import (sample_block_positions_uniform, stack_position_r2frame,
                                                workspace_x_lims_default, workspace_y_lims_default,
                                                sample_block_positions_from_dists)
from manipulation.manipulation_controller import ManipulationController
import numpy as np


def to_canonical_config(config):
    """
    change config to be between -pi and pi for all joints
    """
    for i in range(6):
        while config[i] > np.pi:
            config[i] -= 2 * np.pi
        while config[i] < -np.pi:
            config[i] += 2 * np.pi

    return config


def distribute_blocks_in_positions(block_positions,
                                   robot_controller: ManipulationController,
                                   stack_position=stack_position_r2frame,
                                   is_stack_position_in_ur5e_2_frame=True,
                                   start_height=None):
    """
    distribute blocks from stack positions to given positions
    """
    stack_position_world = robot_controller.gt.point_robot_to_world(robot_controller.robot_name,
                                                                    [*stack_position, 0.])

    if start_height is None:
        start_height = 0.1 + 0.04 * len(block_positions)
    for block_pos in block_positions:
        robot_controller.pick_up(stack_position_world[0], stack_position_world[1], rz=np.pi/2,
                                 start_height=start_height)
        robot_controller.put_down(block_pos[0], block_pos[1], rz=0, start_height=0.15)
        start_height -= 0.04


def ur5e_2_distribute_blocks_in_workspace_uniform(n_blocks,
                                                  robot_controller: ManipulationController,
                                                  ws_lim_x=workspace_x_lims_default,
                                                  ws_lim_y=workspace_y_lims_default,
                                                  stack_position_ur5e_2_frame=stack_position_r2frame,
                                                  min_dist=0.06):
    """
    pickup n_blocks from stack_position and distribute them in the workspace with at least min_dist between them.
    """
    block_positions = sample_block_positions_uniform(n_blocks, ws_lim_x, ws_lim_y, min_dist)
    distribute_blocks_in_positions(block_positions, robot_controller, stack_position_ur5e_2_frame)
    return block_positions


def ur5e_2_distribute_blocks_from_block_positions_dists(blocks_distributions,
                                                        robot_controller: ManipulationController,
                                                        stack_position_ur5e_2_frame=stack_position_r2frame,
                                                        min_distance=0.07):
    block_positions = sample_block_positions_from_dists(blocks_distributions, min_distance)
    distribute_blocks_in_positions(block_positions, robot_controller, stack_position_ur5e_2_frame)
    return block_positions


def ur5e_2_collect_blocks_from_positions(block_positions, robot_controller: ManipulationController,
                                         stack_position_ur5e_2_frame=stack_position_r2frame):
    """
    collect blocks from block_positions and stack them at stack_position
    """
    stack_position_world = robot_controller.gt.point_robot_to_world(robot_controller.robot_name,
                                                                    [*stack_position_ur5e_2_frame, 0.])

    put_down_start_height = 0.1
    for block_pos in block_positions:
        put_down_start_height += 0.04
        robot_controller.pick_up(block_pos[0], block_pos[1], rz=0, start_height=0.15)
        robot_controller.put_down(stack_position_world[0], stack_position_world[1], rz=0,
                                  start_height=put_down_start_height)
