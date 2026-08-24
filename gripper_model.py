import open3d as o3d
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R


class GripperModel:
    """
    Two-finger gripper with local coordinate frame at fingertips.
    Local origin is at the center between the two fingertips.
    X: right, Y: across fingers (opening), Z: forward (pointing away from fingers).
    """

    def __init__(
            self,
            finger_thickness=5.0,
            finger_width=15.0,
            finger_length=100.0,
            opening=150.0,
            base_depth=15.0,
            base_color=[0.2, 0.8, 0.2],
            finger_color=[0.8, 0.2, 0.2],
            axis_size=10.0,
    ):
        self.finger_thickness = finger_thickness
        self.finger_width = finger_width
        self.finger_length = finger_length
        self.opening = opening
        self.base_depth = base_depth
        self.base_color = base_color
        self.finger_color = finger_color
        self.axis_size = axis_size

        # Default pose: local frame aligned with world
        self.pose = np.eye(4)

        # Finger template
        tmpl = o3d.geometry.TriangleMesh.create_box(
            width=self.finger_width,
            height=self.finger_thickness,
            depth=self.finger_length
        )
        tmpl.compute_vertex_normals()
        tmpl.paint_uniform_color(self.finger_color)
        self._finger_template = tmpl

    def transform(self, T: np.ndarray):
        """
        Set the 4x4 pose matrix for the gripper (from local to world).
        """
        assert T.shape == (4, 4)
        self.pose = T

    def _make_finger(self, y_center: float):
        """
        Create one finger mesh, centered at y_center, extending along -Z from origin.
        """
        finger = copy.deepcopy(self._finger_template)
        finger.translate(np.array([
            -self.finger_width / 2.0,
            y_center - self.finger_thickness / 2.0,
            -self.finger_length  # finger extends back from origin (fingertip)
        ]))
        return finger

    def _make_base(self):
        """
        Create base box, behind the fingers along -Z.
        """
        base_height = self.opening + self.finger_thickness*2
        base = o3d.geometry.TriangleMesh.create_box(
            width=self.finger_width,
            height=base_height,
            depth=self.base_depth
        )
        base.compute_vertex_normals()
        base.paint_uniform_color(self.base_color)
        base.translate(np.array([
            -self.finger_width / 2.0,
            -base_height / 2.0,
            -self.finger_length - self.base_depth  # place behind fingers
        ]))
        return base

    def _create_axis_arrows(self):
        """
        Return X/Y/Z arrows (Red/Green/Blue) aligned with current pose.
        """
        origin = self.pose[:3, 3]
        R_matrix = self.pose[:3, :3]
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        arrows = []
        for i in range(3):
            dir_vec = R_matrix[:, i]
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.5,
                cone_radius=1.0,
                cylinder_height=7.0,
                cone_height=3.0
            )
            arrow.paint_uniform_color(colors[i])
            T = np.eye(4)
            T[:3, :3] = align_vector_to_z(dir_vec)
            T[:3, 3] = origin
            arrow.transform(T)
            arrows.append(arrow)
        return arrows

    def get_meshes(self):
        """
        Return list of geometries: [finger1, finger2, base, axis_x, axis_y, axis_z]
        """
        half_open = (self.opening + self.finger_thickness) / 2.0
        f1 = self._make_finger(-half_open)
        f2 = self._make_finger(+half_open)
        base = self._make_base()

        for part in [f1, f2, base]:
            part.transform(self.pose)

        arrows = self._create_axis_arrows()
        return [f1, f2, base] + arrows

    def get_axes(self):
        """
        Return dict of local frame: origin, x, y, z (in world coords).
        """
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return {'origin': t, 'x': R[:, 0], 'y': R[:, 1], 'z': R[:, 2]}

    def get_physical_parameters(self):
        """
        Return physical parameters of the gripper.
        """
        return {
            'finger_thickness': self.finger_thickness,
            'finger_width': self.finger_width,
            'finger_length': self.finger_length,
            'opening': self.opening,
            'base_depth': self.base_depth
        }


def align_vector_to_z(direction: np.ndarray):
    """
    Return rotation matrix aligning Z-axis to the given direction vector.
    """
    z_axis = np.array([0, 0, 1])
    v1 = z_axis / np.linalg.norm(z_axis)
    v2 = direction / np.linalg.norm(direction)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        # 180° rotation around any axis orthogonal to Z
        axis = np.array([1, 0, 0])
        return R.from_rotvec(axis * np.pi).as_matrix()
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R_mat = np.eye(3) + skew + skew @ skew * ((1 - dot) / (np.linalg.norm(cross) ** 2))
    return R_mat


def create_gripper_model(pose=None, **kwargs):
    """
    Create and return a gripper model with specified parameters.

    Args:
        pose (np.ndarray, optional): 4x4 transformation matrix. Defaults to identity.
        **kwargs: Gripper parameters (finger_thickness, finger_width, etc.)

    Returns:
        dict: Contains:
            - 'model': GripperModel instance
            - 'physical_params': Dictionary of physical parameters
            - 'frame_info': Dictionary with origin and axes directions
            - 'meshes': List of gripper meshes (for visualization)
    """
    gripper = GripperModel(**kwargs)
    if pose is not None:
        gripper.transform(pose)

    return {
        'model': gripper,
        'physical_params': gripper.get_physical_parameters(),
        'frame_info': gripper.get_axes(),
        'meshes': gripper.get_meshes()
    }

# if __name__ == '__main__':
#     # 创建默认夹爪
#     gripper_info = create_gripper_model()
#
#     # 创建自定义夹爪
#     custom_gripper = create_gripper_model(
#         finger_length=100,
#         opening=150,
#         finger_color=[0.2, 0.5, 0.8]
#     )
#
#     # 获取信息
#     print(custom_gripper['physical_params'])  # 物理参数
#     print(custom_gripper['frame_info'])  # 坐标系信息
#     o3d.visualization.draw_geometries(custom_gripper['meshes'])  # 可视化