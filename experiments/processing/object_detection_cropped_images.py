import json
from motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
import matplotlib.pyplot as plt
from vision.image_block_position_estimator import ImageBlockPositionEstimator


def load_data():
    metadata = json.load(open("../../vision/images_data_merged_hires/merged_metadata.json", "r"))
    robot_configs = []
    images = []
    depth_images = []
    actual_block_positions = []

    for im_metadata in metadata:
        robot_configs.append(im_metadata["ur5e_1_config"])

        image_path = f"../../vision/images_data_merged_hires/images/{im_metadata['image_rgb']}"
        image = np.load(image_path)
        images.append(image)

        depth_images.append(np.load(f"../../vision/images_data_merged_hires/depth/{im_metadata['image_depth']}"))

    # actual block positions are the same for all images, so we can just take the first one
    actual_block_positions = metadata[0]["block_positions"]

    return robot_configs, images, depth_images, actual_block_positions


def compute_prediction_error(pred, actual):
    """
    Compute the prediction error between the predicted and the actual block positions,
    each prediction is assosiated with nearest actual block position
    :param pred: predicted block positions nx3 array
    :param actual: actual block positions nx3 array
    :return: the mean error
    """
    pred = np.array(pred)
    actual = np.array(actual)

    # there might be less pred than actual. In that case, we will just ignore the extra actuals.
    # compute the error between every pred and actual
    errors = []
    for pred_pos in pred:
        # find the nearest actual block position
        distances = np.linalg.norm(actual[:, :2] - pred_pos[:2], axis=1)
        nearest_idx = np.argmin(distances)
        nearest_actual = actual[nearest_idx]

        error = np.linalg.norm(nearest_actual[:2] - pred_pos[:2])
        errors.append(error)

    return errors


if __name__ == "__main__":
    workspace_limits_x = [-0.9, -0.54]
    workspace_limits_y = [-1.0, -0.55]

    robot_configs, images, depth_images, actual_block_positions = load_data()

    gt = GeometryAndTransforms.build()

    print("Loaded data, beginning prediction.")
    position_estimator = ImageBlockPositionEstimator(workspace_limits_x, workspace_limits_y, gt)

    block_positions, annotations = position_estimator.get_block_positions_depth(images, depth_images, robot_configs)

    actual_block_positions = np.array(actual_block_positions)

    for pred, annotations in zip(block_positions, annotations):
        error = compute_prediction_error(pred, actual_block_positions)
        print(f"error {error}, mean error {np.mean(error)}")

        # create plot with four subplots, 3 for the given annotations and one for the predicted block positions as below
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(annotations[0])
        axs[0, 1].imshow(annotations[1])
        axs[1, 0].imshow(annotations[2])

        pred = np.array(pred)
        axs[1, 1].scatter(actual_block_positions[:, 0], actual_block_positions[:, 1], c='b', s=10)
        axs[1, 1].scatter(pred[:, 0], pred[:, 1], c='r', s=10)
        for i in range(pred.shape[0]):
            axs[1, 1].text(pred[i, 0], pred[i, 1], f'{pred[i, 2]:.3f}', fontsize=9, color='red')
        axs[1, 1].set_xlim(workspace_limits_x)
        axs[1, 1].set_ylim(workspace_limits_y)

        fig.tight_layout()
        plt.show()
