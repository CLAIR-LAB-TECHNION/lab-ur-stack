from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
import matplotlib.pyplot as plt
from lab_ur_stack.vision.image_block_position_estimator import ImageBlockPositionEstimator

from lab_ur_stack.experiments.processing.object_detection_cropped_images import load_data, compute_prediction_error


if __name__ == "__main__":
    workspace_limits_x = [-0.9, -0.54]
    workspace_limits_y = [-1.0, -0.55]

    z_plane_height = -0.015
    z_plane_height += 0.02  # block half height

    robot_configs, images, _, actual_block_positions = load_data()
    gt = GeometryAndTransforms.build()
    print("Loaded data, beginning prediction.")

    position_estimator = ImageBlockPositionEstimator(workspace_limits_x, workspace_limits_y, gt)

    block_positions, annotations = position_estimator.get_block_position_plane_projection(images, robot_configs,
                                                                                         plane_z=z_plane_height)

    actual_block_positions = np.array(actual_block_positions)
    for i, (pred, ann) in enumerate(zip(block_positions, annotations)):
        error = compute_prediction_error(pred, actual_block_positions)
        print(f"--------image {i}--------")
        # print("predicted", pred)
        print(f"error {error},\n mean error {np.mean(error)}")

        # 3 plots: annotated, cropped, and scatter map
        fig, axs = plt.subplots(3, 1, figsize=(5, 10))
        axs[0].imshow(ann[0])
        axs[1].imshow(ann[1])

        pred = np.array(pred)
        axs[2].scatter(pred[:, 0], pred[:, 1], c="r", label="Predicted")
        axs[2].scatter(actual_block_positions[:, 0], actual_block_positions[:, 1], c="b", label="Actual")
        extended_x_lim_for_plot = [workspace_limits_x[0] - 0.1, workspace_limits_x[1] + 0.1]
        extended_y_lim_for_plot = [workspace_limits_y[0] - 0.1, workspace_limits_y[1] + 0.1]
        axs[2].set_xlim(extended_x_lim_for_plot)
        axs[2].set_ylim(extended_y_lim_for_plot)

        plt.show()

