import numpy as np
import pyrender
import trimesh
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.camera.configurations_and_params import color_fx, color_fy, color_ppx, color_ppy


class PyrenderRenderer:
    box_size = [0.04, 0.04, 0.04]

    def __init__(self, gt: GeometryAndTransforms = None):
        self.gt = gt if gt is not None else GeometryAndTransforms.build()
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0, 1.0])
        self.renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)

        camera = pyrender.IntrinsicsCamera(color_fx, color_fy, color_ppx, color_ppy)
        self.scene.add(camera)

        point_light_positions = [
            [2.0, 2.0, 3.0],
            [-2.0, -2.0, 3.0],
            [2.0, -2.0, 3.0],
            [-2.0, 2.0, 3.0]]
        for pos in point_light_positions:
            light_pose = np.eye(4)
            light_pose[:3, 3] = pos
            light = pyrender.PointLight(color=np.ones(3), intensity=500.0)
            self.scene.add(light, pose=light_pose)

    def set_boxes_positions(self, positions):
        # remove all boxes from the scene
        for node in self.scene.mesh_nodes:
            self.scene.remove_node(node)

        for coord in positions:
            box_mesh = trimesh.creation.box(extents=self.box_size)
            box_mesh.visual.face_colors = [0, 255, 0, 255]
            box_mesh.apply_translation(coord)
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 1.0, 0.0, 1.0])
            box = pyrender.Mesh.from_trimesh(box_mesh, material=material)
            self.scene.add(box)

    def render_from_robot_config(self, robot_name, robot_config):
        camera_pose = self.gt.camera_to_world_transform(robot_name, robot_config)
        camera_pose = self.gt.se3_to_4x4(camera_pose)
        color, depth = self.render_from_camera_pose(camera_pose)
        return color, depth

    def render_from_camera_pose(self, camera_pose_4x4):
        # update camera pose:
        camera = self.scene.main_camera_node
        camera.matrix = self.camera_pose_to_openGL(camera_pose_4x4)
        return self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)

    def camera_pose_to_openGL(self, camera_pose):
        # y and z are in the opposite directions
        opengl_camera_transform = np.array([[1, 0, 0, 0],
                                            [0, -1, 0, 0],
                                            [0, 0, -1, 0],
                                            [0, 0, 0, 1]])
        return camera_pose @ opengl_camera_transform

