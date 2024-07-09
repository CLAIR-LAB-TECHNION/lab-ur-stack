import cv2

from vision.utils import crop_workspace
from vision.object_detection import ObjectDetection
import json
from motion_planning.geometry_and_transforms import GeometryAndTransforms
import numpy as np
import matplotlib.pyplot as plt
from camera.realsense_camera import project_color_pixel_to_depth_pixel
from camera.configurations_and_params import color_camera_intrinsic_matrix


class ImageBlockPositionEstimator:
    def __init__(self, workspace_limits_x, workspace_limits_y, gt: GeometryAndTransforms, robot_name='ur5e_1'):
        self.workspace_limits_x = workspace_limits_x
        self.workspace_limits_y = workspace_limits_y
        self.gt = gt
        self.detector = ObjectDetection()
        self.robot_name = robot_name

    def bboxes_cropped_to_orig(self, bboxes, xyxy):
        bboxes_orig = bboxes.cpu().numpy().copy()
        bboxes_orig[:, 0] += xyxy[0]
        bboxes_orig[:, 1] += xyxy[1]
        bboxes_orig[:, 2] += xyxy[0]
        bboxes_orig[:, 3] += xyxy[1]
        return bboxes_orig

    def points_image_to_camera_frame(self, points_image_xy, z_depth):
        """
        Transform points from image coordinates to camera frame coordinates
        :param points_image_xy: points in image coordinates nx2
        :param z_depth: depth of the points nx1
        :return: points in camera frame coordinates nx3
        """
        fx = color_camera_intrinsic_matrix[0, 0]
        fy = color_camera_intrinsic_matrix[1, 1]
        cx = color_camera_intrinsic_matrix[0, 2]
        cy = color_camera_intrinsic_matrix[1, 2]

        x_camera = (points_image_xy[:, 0] - cx) * z_depth / fx
        y_camera = (points_image_xy[:, 1] - cy) * z_depth / fy
        points_camera_frame = np.vstack((x_camera, y_camera, z_depth)).T

        return points_camera_frame

    def get_z_depth_mean(self, points_color_image_xy, depth_image, windows_sizes):
        """
        Get mean depth of the points in the window around the points. discard points with depth 0.
        This method project the point in from image to depth
        :param points_color_image_xy: the point in color image coordinates
        :param depth_image:
        :param windows_sizes:
        :return: mean depth in windows around points, -1 if no depth around the point or invalid points
            and also the xyxy of the windows used for depth computation in the depth image
        """
        points_in_depth = [project_color_pixel_to_depth_pixel(point, depth_image) for point in points_color_image_xy]

        depths = []
        windows_xyxy = []
        for point, win_size in zip(points_in_depth, windows_sizes):
            if point[0] < 0 or point[1] < 0 or point[0] >= depth_image.shape[1] or point[1] >= depth_image.shape[0]:
                depths.append(-1)
                continue

            # make sure window is within the image
            x_min = max(0, point[0] - win_size[0])
            x_max = min(depth_image.shape[1], point[0] + win_size[0])
            y_min = max(0, point[1] - win_size[1])
            y_max = min(depth_image.shape[0], point[1] + win_size[1])

            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
            windows_xyxy.append((x_min, x_max, y_min, y_max))

            window = depth_image[y_min:y_max, x_min:x_max]
            if np.all(window <= 0):
                depths.append(-1)
                continue

            depths.append(np.mean(window[window > 0]))

        return depths, windows_xyxy

    def get_block_positions_depth(self, images, depths, robot_configurations, return_annotations=True,
                                  z_offset=0.02, detect_on_cropped=True):
        """
        Get block positions from images and depth images. can be single image or batch of images
        :param images: ims to get block positions from
        :param depths: depths to get block positions from
        :param robot_configurations: robot configurations for each image
        :param return_annotations: whether to return annotated images
        :param z_offset: offset from the detected depth to the block position, this is the length of the block,
            since depth is measured to face and we want to get the center of the block
        :param detect_on_cropped: if True, detect objects on cropped images, if False, detect on full images
        :return: list of detected block positions in world coordinates for each image
            (or single list if single image is given)
            if return_annotations is True, another or list of tuples of images will be returned.
            The tuple contains (annotated_cropped, annotated, depth_with_windows)
            annotated cropped is the image that was used for od with detections, classes and probabilities,
            annotated is the original image with just detections and center points,
            depth_with_windows is the depth image with windows around the center points that contains all the pixels
                that was used for depth computation
        """
        if len(np.array(images).shape) == 3:
            images = [images]
            robot_configurations = [robot_configurations]
            depths = [depths]

        bboxes, bboxes_centers, bbox_sizes, results, annotated_cropped, annotated =\
            self.get_bboxes(detect_on_cropped, images, robot_configurations, return_annotations)

        depth_with_windows = []
        estimated_z_depths = []
        for bbox_centers_curr, bbox_sizes_curr, depth_im in zip(bboxes_centers, bbox_sizes, depths):
            # take at least 3x3 pixels, at most 10x10 pixels, and as default half of the bbox size,
            # separately for x and y
            bbox_sizes_curr = np.array(bbox_sizes_curr)
            win_sizes_x = np.clip(np.ceil(bbox_sizes_curr[:, 0] / 2), 3, 10)
            win_sizes_y = np.clip(np.ceil(bbox_sizes_curr[:, 1] / 2), 3, 10)
            windows_sizes = np.vstack((win_sizes_x, win_sizes_y)).T

            estimated_z_depths_curr, windows_xyxy = self.get_z_depth_mean(bbox_centers_curr, depth_im, windows_sizes)
            estimated_z_depths_curr = np.array(estimated_z_depths_curr) + z_offset
            estimated_z_depths.append(estimated_z_depths_curr)

            if return_annotations:
                max_depth_for_plot = 3
                depth_with_windows_curr = depth_im.copy()
                depth_with_windows_curr = np.clip(depth_with_windows_curr, 0, max_depth_for_plot)
                depth_with_windows_curr = ((depth_with_windows_curr / max_depth_for_plot) * 255).astype(np.uint8)
                depth_with_windows_curr = cv2.cvtColor(depth_with_windows_curr, cv2.COLOR_GRAY2RGB)
                for win_xyxy in windows_xyxy:
                    depth_with_windows_curr = cv2.rectangle(depth_with_windows_curr, (win_xyxy[0], win_xyxy[2]),
                                                            (win_xyxy[1], win_xyxy[3]), (0, 255, 0), 1)
                depth_with_windows.append(depth_with_windows_curr)

        block_positions_camera_frame = []
        for bbox_centers_curr, depth in zip(bboxes_centers, estimated_z_depths):
            block_positions_camera_frame.append(self.points_image_to_camera_frame(bbox_centers_curr, depth))

        block_positions_world = []
        for block_positions_camera_frame_curr, robot_config in zip(block_positions_camera_frame, robot_configurations):
            # we do point by point, this method is not validated as vectorized yet, need to do that
            curr_block_positions_world = []
            for bpos_cam in block_positions_camera_frame_curr:
                block_position_world = self.gt.point_camera_to_world(bpos_cam, self.robot_name, robot_config)
                curr_block_positions_world.append(block_position_world)
            block_positions_world.append(curr_block_positions_world)

        # if single image, return single list
        if len(block_positions_world) == 1:
            if return_annotations:
                return block_positions_world[0], (annotated_cropped[0], annotated[0], depth_with_windows[0])
            return block_positions_world[0]

        if return_annotations:
            # convert to list of tuples
            return block_positions_world, list(zip(annotated_cropped, annotated, depth_with_windows))
        return block_positions_world

    def get_block_position_plane_projection(self, images, robot_configurations, plane_z=-0.01,
                                            return_annotations=True, detect_on_cropped=True):
        """
        Get block positions from images, without depth. To get the missing information that we get from depth,
        we assume that the block is on a known plane (e.g. table) with known z coordinate, and we project the
        detected block position to that plane.

        parameters  are similar to get_block_positions_depth

        :param images:
        :param robot_configurations:
        :param plane_z: the z coordinate of the plane in the world frame
        :param return_annotations:
        :param detect_on_cropped:
        :return: list of detected block positions for each image and tuple of annotations if return_annotations is True
            the tuple contains (annotated_cropped, annotated) similar to get_block_positions_depth
             (just without the depth)
        """
        if len(np.array(images).shape) == 3:
            images = [images]
            robot_configurations = [robot_configurations]

        # we will only fill these if return_annotations is True
        bboxes, bboxes_centers, _, results, annotated_cropped, annotated =\
            self.get_bboxes(detect_on_cropped, images, robot_configurations, return_annotations)

        block_positions_world = []

        for bbox_centers_curr, robot_config in zip(bboxes_centers, robot_configurations):
            block_positions_world_curr = []

            # Assume depth is unit, we only create direction vectors:
            z_depth = np.ones_like(bbox_centers_curr[:, 0])

            # Transform image points to camera frame using intrinsic parameters
            points_camera_frame = self.points_image_to_camera_frame(bbox_centers_curr, z_depth)

            # Normalize the points to get direction vectors
            directions_camera_frame = points_camera_frame / np.linalg.norm(points_camera_frame, axis=1)[:, None]

            # Get transformation from camera to world frame
            transform_camera_to_world = self.gt.camera_to_world_transform(self.robot_name, robot_config)
            transform_camera_to_world = self.gt.se3_to_4x4(transform_camera_to_world)
            R_cam_to_world = transform_camera_to_world[:3, :3]
            t_cam_to_world = transform_camera_to_world[:3, 3]

            for direction in directions_camera_frame:
                direction_world = R_cam_to_world @ direction

                if direction_world[2] == 0:
                    # if direction is parallel to the plane, we can't find the intersection
                    continue

                # define a line in space that goes through the camera center and in the direction
                # of the detected block
                # the line is defined as x = t + lambda * direction, where x is the point on the line,

                # we want to find the lambda that makes the z coordinate of the point on the line equal to plane_z
                # so we solve for lambda in the equation t_cam_to_world[2] + lambda * direction_world[2] = plane_z

                lambda_param = (plane_z - t_cam_to_world[2]) / direction_world[2]
                pos_world = t_cam_to_world + lambda_param * direction_world

                block_positions_world_curr.append(pos_world)

            block_positions_world.append(block_positions_world_curr)

        # If single image, return single list
        if len(block_positions_world) == 1:
            if return_annotations:
                return block_positions_world[0], (annotated_cropped[0], annotated[0])
            return block_positions_world[0]

        if return_annotations:
            # Convert to list of tuples
            return block_positions_world, list(zip(annotated_cropped, annotated))
        return block_positions_world

    def get_bboxes(self, detect_on_cropped, images, robot_configurations, return_annotations):

        # we will only fill these if return_annotations is True:
        annotated_cropped = []
        annotated = []


        if detect_on_cropped:
            cropped_images = []
            cropped_images_xyxy = []
            for image, robot_config in zip(images, robot_configurations):
                cropped_image, xyxy = crop_workspace(image, robot_config, self.gt, self.workspace_limits_x,
                                                     self.workspace_limits_y)
                cropped_images.append(cropped_image)
                cropped_images_xyxy.append(xyxy)

            bboxes_in_cropped, _, results = self.detector.detect_objects(cropped_images)
            bboxes = [self.bboxes_cropped_to_orig(bboxes, xyxy)
                      for bboxes, xyxy in zip(bboxes_in_cropped, cropped_images_xyxy)]
        else:
            bboxes, _, results = self.detector.detect_objects(images)

        bboxes_centers = [(bbox[:, :2] + bbox[:, 2:]) / 2 for bbox in bboxes]
        bbox_sizes = [(bbox[:, 2:] - bbox[:, :2]) for bbox in bboxes]

        if return_annotations:
            for res, bboxes_centers_curr, bboxes_curr_im, im in zip(results, bboxes_centers, bboxes, images):
                annotated_cropped.append(res.plot())
                annotated_image = im.copy()
                for bbox_center in bboxes_centers_curr:
                    annotated_image = cv2.circle(annotated_image, tuple(bbox_center.astype(int)), 6, (256, 0, 0), -1)
                annotated.append(annotated_image)

        return bboxes, bboxes_centers, bbox_sizes, results, annotated_cropped, annotated

