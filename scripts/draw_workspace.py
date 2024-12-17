import typer

from lab_ur_stack.manipulation.manipulation_controller import ManipulationController
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_2

app = typer.Typer()

workspace_x_lims = [-0.9, -0.54]
workspace_y_lims = [-1.0, -0.55]


@app.command()
def main(robot_name="ur5e_2"):
    rob = ManipulationController.build_from_robot_name_and_ip(ur5e_2["ip"], ur5e_2["name"])

    four_points = [[workspace_x_lims[0], workspace_y_lims[0]],
                   [workspace_x_lims[0], workspace_y_lims[1]],
                   [workspace_x_lims[1], workspace_y_lims[1]],
                   [workspace_x_lims[1], workspace_y_lims[0]]]

    for p in four_points:
        rob.plan_and_move_to_xyzrz(p[0], p[1], 0.01, 0)


if __name__ == "__main__":
    app()
