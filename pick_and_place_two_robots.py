from typing import List
import typer
from motion_planning.motion_planner import MotionPlanner
from motion_planning.geometry_and_transforms import GeometryAndTransforms
from manipulation_controller import ManipulationController
from robot_inteface.robots_metadata import ur5e_1, ur5e_2

r1_clearance_config = [0.7600, -1.8429, 0.8419, -1.3752, -1.5739, -2.3080]
r2_clearance_config = [0.7600, -1.8429, 0.8419, -1.3752, -1.5739, -2.3080]
joint_pick_and_drop_position = [-0.49844, -0.69935]
robot_1_pick_and_drop_position = [-0.4, -0.2]
robot_2_default_pick_and_drop_position = [-0.54105, -0.95301]

app = typer.Typer()


@app.command(
    context_settings={"ignore_unknown_options": True})
def main(robot_1_pickup_x: float = robot_2_default_pick_and_drop_position[0],
         robot_1_pickup_y: float = robot_2_default_pick_and_drop_position[1],):

    motion_planner = MotionPlanner()
    gt = GeometryAndTransforms.from_motion_planner(motion_planner)

    r1_controller = ManipulationController(ur5e_1["ip"], ur5e_1["name"], motion_planner, gt)
    r2_controller = ManipulationController(ur5e_2["ip"], ur5e_2["name"], motion_planner, gt)
    # r1_controller.speed *= 0.6
    # r1_controller.acceleration *= 0.6
    # r2_controller.speed *= 0.6
    # r2_controller.acceleration *= 0.6

    # motion plan and move to clearance config:
    r1_controller.plan_and_moveJ(r1_clearance_config, speed=1., acceleration=1.)
    r2_controller.plan_and_moveJ(r2_clearance_config, speed=1., acceleration=1.)

    # robot 2 picks up and drops at joint position:
    r2_controller.pick_up(robot_1_pickup_x, robot_1_pickup_y, 0)
    r2_controller.put_down(joint_pick_and_drop_position[0], joint_pick_and_drop_position[1], 0)

    r2_controller.moveJ(r2_clearance_config, speed=1., acceleration=1.)
    #TODO: need to implement that in a better way:
    motion_planner.ur5e_2.setConfig([0, *r2_clearance_config, 0])

    # robot 1 picks up from joint position and drops at robot's1 position:
    r1_controller.pick_up(joint_pick_and_drop_position[0], joint_pick_and_drop_position[1], 0)
    r1_controller.put_down(robot_1_pick_and_drop_position[0], robot_1_pick_and_drop_position[1], 0)

    r1_controller.moveJ(r1_clearance_config, speed=1., acceleration=1.)
    #TODO: need to implement that in a better way:
    motion_planner.ur5e_1.setConfig([0, *r1_clearance_config, 0])


if __name__ == "__main__":
    app()

