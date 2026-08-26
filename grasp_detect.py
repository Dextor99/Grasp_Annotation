import sys
import math
import open3d as o3d
import numpy as np
import copy
from scipy.spatial import cKDTree
from cloud_process import frames_process
from gripper_model import create_gripper_model
from profiling import active_profiler, profiled
from depth_sampling import generate_depth_samples
from depth_profile import DepthProfiler
from opening_profile import OpeningProfiler

def generate_cylinder_sections(origin,pcd_points, frame, cyl_radius=75.0, offset=100.0, height=1.0):
    """
    在指定 frame 上生成两个圆柱面截面（一个接触面，一个延申面）

    参数:
        pcd_points: ndarray (N, 3)，点云数据
        frame: dict，包含 origin, x_axis, y_axis, z_axis
        cyl_radius: 半径 (mm)
        offset: 第二截面距离 (mm)
        height: 圆柱厚度 (mm)

    返回:
        cyl0, cyl1: 两个 o3d.geometry.TriangleMesh 圆柱对象
        center0, center1: 圆柱中心位置
    """
    # # origin = frame['origin']
    # origin = frame['origin'] + frame['x_axis'] - frame['y_axis']*10
    z_axis = frame['z_axis']
    z_axis = z_axis / np.linalg.norm(z_axis)

    # 所有点向平面中心的向量
    v_all = pcd_points - origin
    t_all = v_all @ z_axis  # 点到平面的距离投影（沿z轴）
    radial = np.linalg.norm(v_all - np.outer(t_all, z_axis), axis=1)
    mask = radial <= cyl_radius

    ts = t_all[mask]
    if len(ts) == 0:
        print("No intersection found for this direction.")
        return None, None, None, None

    t0 = np.min(ts)
    t1 = t0 + offset

    center0 = origin + z_axis * t0
    center1 = origin + z_axis * t1

    # 创建两个圆柱面（默认方向沿 Z+）
    cyl0 = o3d.geometry.TriangleMesh.create_cylinder(radius=cyl_radius, height=height)
    cyl1 = copy.deepcopy(cyl0)

    # 旋转至 z_axis 方向
    default_z = np.array([0, 0, 1.0])
    axis = np.cross(default_z, z_axis)
    if np.linalg.norm(axis) < 1e-8:
        R = np.eye(3)
    else:
        angle = math.acos(np.clip(np.dot(default_z, z_axis), -1.0, 1.0))
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle)

    cyl0.rotate(R, center=(0, 0, 0))
    cyl1.rotate(R, center=(0, 0, 0))

    # 平移到最终位置（注意圆柱默认底面在 Z=0, 顶面在 Z=height）
    cyl0.translate(center0 - z_axis * height / 2)
    cyl1.translate(center1 - z_axis * height / 2)

    return cyl0, cyl1, center0, center1


def transform_gripper_to_frame(gripper_meshes, frame, origin, T_object_world):
    """
    将夹爪从局部原始坐标，变换到指定 frame 坐标系，并可计算其在物体坐标系下的姿态

    参数:
        gripper_meshes: 原始 gripper TriangleMesh 列表
        frame: 包含 x/y/z_axis 的字典
        origin: 夹爪在世界坐标系下放置的原点
        T_object_world: 物体 → 世界 坐标系变换 (4x4)，如果提供将返回夹爪在物体坐标系下的变换

    返回:
        transformed_meshes: 已变换的 gripper 模型
        T_gripper_world: 夹爪在世界坐标系下的 4x4 变换矩阵
        T_gripper_object: 夹爪在物体坐标系下的 4x4 变换矩阵（如果传入 T_object_world）
        axes_matrix: gripper 的三个方向向量列向量组成的矩阵
        origin: 原点坐标
    """
    x_axis = frame['x_axis']
    y_axis = frame['y_axis']
    z_axis = frame['z_axis']
    axes_matrix = np.column_stack((x_axis, y_axis, z_axis))  # shape (3, 3)

    T_gripper_world = np.eye(4)
    T_gripper_world[:3, :3] = axes_matrix
    T_gripper_world[:3, 3] = origin

    transformed_meshes = []
    for mesh in gripper_meshes:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(T_gripper_world)
        transformed_meshes.append(mesh_copy)

    if T_object_world is not None:
        T_world_object = np.linalg.inv(T_object_world)
        T_gripper_object = T_world_object @ T_gripper_world
    else:
        T_gripper_object = None

    return transformed_meshes, T_gripper_world, T_gripper_object, axes_matrix, origin

#验证夹爪坐标系转换矩阵
def transform_gripper_via_object_frame(gripper_meshes,
                                       T_gripper_object,
                                       T_object_world):
    """
    将在世界坐标原点的夹爪（开口方向为默认Z轴）按如下步骤变换：
    世界系原点 → 物体坐标系 → 夹爪目标位姿
    参数：
        gripper_meshes: list of open3d.geometry.TriangleMesh
            夹爪的 mesh 模型列表（默认位于世界系原点）
        T_gripper_object: gripper 在物体坐标系下的姿态 (4x4)
        T_object_world: 物体坐标系在世界坐标系下的姿态 (4x4)

    返回：
        transformed_gripper_meshes: list of meshes 变换后的夹爪（在世界坐标系下）
    """
    # 组合变换：夹爪最终在世界系下的姿态
    T_gripper_world = T_object_world @ T_gripper_object

    # 对所有 gripper mesh 执行同样的变换
    transformed_gripper_meshes = []
    for mesh in gripper_meshes:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(T_gripper_world)
        transformed_gripper_meshes.append(mesh_copy)

    return transformed_gripper_meshes

#############生成夹爪系列
def generate_gripper_variants(
    base_gripper,
    T_object_world: np.ndarray,
    T_gripper_object: np.ndarray,
    step_deg=5,
    max_deg=180,
    step_open=5,
    max_open=150
):
    variants = []
    idx = 0

    # 始终在物体坐标系下做角度变化
    R_gripper = T_gripper_object[:3, :3]
    t_gripper = T_gripper_object[:3, 3]

    for angle_deg in range(0, max_deg + 1, step_deg):
        theta_rad = np.deg2rad(angle_deg)

        # 局部 Z 轴旋转
        Rz = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad),  np.cos(theta_rad), 0],
            [0, 0, 1]
        ])

        R_new = R_gripper @ Rz
        T_gripper_object_i = np.eye(4)
        T_gripper_object_i[:3, :3] = R_new
        T_gripper_object_i[:3, 3] = t_gripper

        # 最终 mesh 要变换到世界系下显示
        T_gripper_world = T_object_world @ T_gripper_object_i

        for opening in range(0, max_open + 1, step_open):
            gripper = copy.deepcopy(base_gripper)
            gripper.opening = opening
            gripper.transform(T_gripper_world)
            meshes = gripper.get_meshes()

            variants.append({
                'id': idx,
                'angle_deg': angle_deg,
                'opening': opening,
                'T_gripper_object': T_gripper_object_i.copy(),  # 保留物体坐标系下的姿态
                'meshes': meshes,
                'origin': t_gripper.copy(),                     # ✅ 原点仍然是物体坐标系下
                'model': gripper,
            })
            idx += 1

    return variants

###############生成沿Z轴平移的夹爪集合
def slide_gripper_along_z(
    gripper_variants,
    T_object_world,
    step_mm=4,
    max_distance=100,
    depths=None,
):
    moved_grippers = []
    idx = 0

    if depths is None:
        depths = range(0, max_distance + 1, step_mm)
    for depth_id, d in enumerate(depths):
        for variant in gripper_variants:
            base_pose = variant['T_gripper_object']
            base_model = variant['model']
            opening = variant['opening']
            angle_deg = variant['angle_deg']
            origin = variant['origin']

            # 平移矩阵（沿局部Z轴）
            T_translation = np.eye(4)
            T_translation[:3, 3] = np.array([0, 0, d])

            # 在物体坐标系下平移
            T_gripper_object_shifted = base_pose @ T_translation

            # ✅ 最终在世界坐标系下的位姿
            T_gripper_world_shifted = T_object_world @ T_gripper_object_shifted

            # ✅ 使用未变换的 base_model，再应用新的世界坐标变换
            gripper = copy.deepcopy(base_model)
            gripper.transform(T_gripper_world_shifted)

            moved_grippers.append({
                'id': idx,
                'base_id': variant['id'],
                'depth_id': depth_id,
                'angle_deg': angle_deg,
                'opening': opening,
                'depth': d,
                'T_object_world':T_object_world,
                'T_gripper_object': T_gripper_object_shifted.copy(),  # 在物体坐标系下
                'meshes': gripper.get_meshes(),
                'origin': T_gripper_object_shifted[:3, 3].copy(),      # 原点仍为物体坐标系下
                'model': gripper,
            })

            idx += 1

    return moved_grippers

#####################碰撞检测
def aabb_overlaps(min_a, max_a, min_b, max_b, margin=0.0):
    """Return whether two axis-aligned boxes overlap after expanding both by margin."""
    min_a = np.asarray(min_a, dtype=float)
    max_a = np.asarray(max_a, dtype=float)
    min_b = np.asarray(min_b, dtype=float)
    max_b = np.asarray(max_b, dtype=float)
    if any(value.shape != (3,) for value in (min_a, max_a, min_b, max_b)):
        raise ValueError("AABB bounds must have shape (3,)")
    return bool(np.all(min_a <= max_b + margin) and np.all(min_b <= max_a + margin))


class CollisionIndex:
    """Reusable point-cloud index for collision broad-phase queries."""

    def __init__(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("collision index requires a non-empty (N, 3) point array")
        if not np.isfinite(points).all():
            raise ValueError("collision index points must be finite")
        self.points = points
        self.tree = cKDTree(points)
        self.min_bound = points.min(axis=0)
        self.max_bound = points.max(axis=0)

    @classmethod
    def from_point_cloud(cls, point_cloud):
        return cls(np.asarray(point_cloud.points))

    def query_aabb(self, min_bound, max_bound):
        """Return original point indices inside an AABB using the cached KDTree."""
        min_bound = np.asarray(min_bound, dtype=float)
        max_bound = np.asarray(max_bound, dtype=float)
        center = (min_bound + max_bound) / 2.0
        radius = float(np.linalg.norm(max_bound - min_bound) / 2.0)
        candidates = self.tree.query_ball_point(center, radius)
        if not candidates:
            return np.empty(0, dtype=np.int64)
        indices = np.asarray(candidates, dtype=np.int64)
        local = self.points[indices]
        inside = np.all((local >= min_bound) & (local <= max_bound), axis=1)
        return np.sort(indices[inside])


def check_collision(gripper_meshes, point_cloud, threshold=2.0, collision_index=None):
    """
    检测夹爪与点云是否碰撞
    参数:
        gripper_meshes: 夹爪网格列表 [finger1, finger2, base, axis]
        point_cloud: 目标点云
        threshold: 碰撞判定阈值(mm)
    返回:
        bool: True表示有碰撞，False表示无碰撞
    """
    if collision_index is None:
        collision_index = CollisionIndex.from_point_cloud(point_cloud)

    # 合并夹爪所有网格
    combined_mesh = gripper_meshes[0] + gripper_meshes[1] + gripper_meshes[2]

    gripper_min = np.asarray(combined_mesh.get_min_bound(), dtype=float)
    gripper_max = np.asarray(combined_mesh.get_max_bound(), dtype=float)
    if not np.isfinite(gripper_min).all() or not np.isfinite(gripper_max).all():
        return True

    # Broad phase: no object point can collide outside the expanded gripper AABB.
    local_min = gripper_min - threshold
    local_max = gripper_max + threshold
    if not aabb_overlaps(local_min, local_max, collision_index.min_bound, collision_index.max_bound):
        return False
    local_indices = collision_index.query_aabb(local_min, local_max)
    if len(local_indices) == 0:
        return False

    # 精确阶段仍使用原有 1000 点夹爪采样，但只查询局部物体点。
    gripper_pcd = combined_mesh.sample_points_uniformly(number_of_points=1000)
    gripper_points = np.asarray(gripper_pcd.points)
    if len(gripper_points) == 0 or not np.isfinite(gripper_points).all():
        return True
    gripper_tree = cKDTree(gripper_points)
    distances, _ = gripper_tree.query(collision_index.points[local_indices], k=1)
    return bool(np.any(distances < threshold))

def filter_collision_free_grippers(gripper_list, point_cloud, threshold=2.0):
    """
    对夹爪列表进行碰撞检测，筛选无碰撞夹爪。

    参数：
        gripper_list: 包含多个夹爪数据的 list，每个元素包含 'meshes'、'opening'、'angle_deg' 等字段
        point_cloud: Open3D 点云对象
        threshold: 判定为碰撞的距离阈值（mm）

    返回：
        filtered_list: 无碰撞的夹爪列表（字段保持不变）
    """
    collision_index = CollisionIndex.from_point_cloud(point_cloud)
    filtered_list = []
    total = len(gripper_list)
    for i, gripper in enumerate(gripper_list):
        if not check_collision(gripper['meshes'], point_cloud, threshold, collision_index=collision_index):
            filtered_list.append(gripper)
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"检测进度：{i+1}/{total} 个夹爪")

    print(f"原始夹爪数量：{total}")
    print(f"无碰撞夹爪数量：{len(filtered_list)}")
    return filtered_list


##########################保留每个depth和旋转角度下，开口最小的一个夹爪
# Structural invalidity is defined as opening == 0 OR depth == 0.
def filter_structurally_valid_grippers(gripper_list):
    """Remove candidates invalid by the method definition before collision checks."""
    return [
        gripper for gripper in gripper_list
        if gripper.get("opening", 0) > 0 and gripper.get("depth", 0) > 0
    ]


def filter_by_min_opening_per_depth_angle(gripper_list):
    """
    在每个 depth 和 angle_deg 下保留 opening 最小的夹爪。
    会先排除 opening == 0 且 depth == 0 的夹爪。

    参数：
        gripper_list: list，每个元素是 dict，包含 'depth'、'angle_deg'、'opening' 等字段

    返回：
        filtered: list，只保留每组 opening 最小的夹爪
    """
    from collections import defaultdict

    # 1. 先过滤掉 opening==0 且 depth==0 的夹爪
    # Structural rule: opening == 0 OR depth == 0 is invalid.
    cleaned = [
        g for g in gripper_list
        if not (g['opening'] == 0 or g['depth'] == 0)
    ]

    grouped = defaultdict(list)

    # 2. 分组
    for g in cleaned:
        key = (g['depth'], g['angle_deg'])
        grouped[key].append(g)

    filtered = []

    # 3. 每组中找 opening 最小的
    for key, group in grouped.items():
        min_gripper = min(group, key=lambda x: x['opening'])
        filtered.append(min_gripper)

    print(f"原始夹爪数：{len(gripper_list)}")
    print(f"清理后夹爪数：{len(cleaned)}")
    print(f"每层每角度最优夹爪数：{len(filtered)}")
    return filtered

########################去掉开口为0 且二指之间无物体点云的夹爪
def is_point_between_fingers(point, T_gripper_world, opening, finger_length, finger_thickness=15):
    """
    检查某点是否落在两指之间的包围区域（简化为盒子）。
    夹爪坐标系假设：
        - X 轴：从左指指心 -> 右指指心
        - Z 轴：指向物体，平移方向
        - Y 轴：指尖方向
    """
    # 转换点到夹爪坐标系
    p_local = T_gripper_world[:3, :3].T @ (point - T_gripper_world[:3, 3])
    x, y, z = p_local[:3]

    # 假设指间宽度为 opening，中间夹持区域 ±thickness/2，前后为 finger_length
    in_x = ((-finger_thickness / 2)-1 <= x <= (finger_thickness / 2)+1)
    in_y = (-opening / 2 <= y <= opening / 2)
    in_z = (-finger_length <= z <= -1)
    return in_x and in_y and in_z

def filter_grippers_with_object_between_fingers(T_object_world,grippers, point_cloud, finger_length=100):
    """
    过滤夹爪之间有物体点云的夹爪。

    参数：
        grippers: list，夹爪列表（必须含有 opening 和 T_object_gripper）
        point_cloud: o3d.geometry.PointCloud，目标物体点云
        finger_length: 夹爪指长

    返回：
        filtered_grippers: list，符合要求的夹爪
    """
    filtered = []

    points = np.asarray(point_cloud.points)

    for g in grippers:
        opening = g['opening']
        T_gripper_object = g['T_gripper_object']
        T_gripper_world = T_object_world @ T_gripper_object

        found = False
        for pt in points:
            if is_point_between_fingers(pt, T_gripper_world, opening, finger_length):
                found = True
                break

        if found:
            filtered.append(g)

    print(f"原始数量: {len(grippers)}")
    print(f"保留夹爪（开口非0 + 有点云）数量: {len(filtered)}")
    return filtered

###########################将夹爪角点转换到世界坐标系中
def create_colored_sphere(center, radius=2.0, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def get_box_lineset(box_corners_world, color=[1, 0, 0]):
    """
    用于把8个角点连接成线框盒子（立方体）
    """
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 上方
        [4, 5], [5, 6], [6, 7], [7, 4],  # 下方
        [0, 4], [1, 5], [2, 6], [3, 7],  # 竖向连接
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(box_corners_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.paint_uniform_color(color)
    return line_set


def visualize_gripper_box_and_pointcloud(box_corners_world, pcd, sphere_radius=2.0):
    """
    显示夹爪8个角点、包围框线段、以及物体点云
    """
    # 红色角点球
    spheres = [create_colored_sphere(p, sphere_radius, color=[1, 0, 0]) for p in box_corners_world]
    # 线框包围盒
    box_lines = get_box_lineset(box_corners_world, color=[1, 0, 0])
    # 点云颜色设为灰色
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    o3d.visualization.draw_geometries([pcd, box_lines] + spheres)
def get_transformed_box_corners(T_gripper_world, opening, finger_width, finger_length):
    """夹爪坐标系下的 8 个点，变换到世界坐标系"""
    fw = finger_width / 2
    op = opening / 2
    fl = finger_length

    # 局部坐标系下的角点 (8, 3)
    corners_local = np.array([
        [-fw,  op,  0],
        [-fw, -op,  0],
        [ fw, -op,  0],
        [ fw,  op,  0],
        [-fw,  op, -fl],
        [-fw, -op, -fl],
        [ fw, -op, -fl],
        [ fw,  op, -fl],
    ])

    # 变换到世界坐标系
    R = T_gripper_world[:3, :3]
    t = T_gripper_world[:3, 3]
    corners_world = (R @ corners_local.T).T + t
    return corners_world  # (8, 3)

def is_points_in_convex_box(points_world, box_corners_world):
    """
    判断哪些点在 8 个角点构成的**六面体长方体**中。
    算法：构造 6 个面，点必须在所有面的“内部”（法向量朝外，点在法向内侧）
    """
    def get_face_normal(p0, p1, p2):
        return np.cross(p1 - p0, p2 - p0)

    faces = [
        [0, 1, 2, 3],  # 顶面
        [4, 5, 6, 7],  # 底面
        [0, 4, 7, 3],  # 前面
        [1, 5, 6, 2],  # 后面
        [0, 1, 5, 4],  # 左面
        [3, 2, 6, 7],  # 右面
    ]

    mask = np.ones(len(points_world), dtype=bool)
    for face in faces:
        p0, p1, p2 = [box_corners_world[i] for i in face[:3]]
        normal = get_face_normal(p0, p1, p2)
        normal = normal / np.linalg.norm(normal)
        # 点到平面：dot((x - p0), n) <= 0 表示在法向的“内侧”
        dots = (points_world - p0) @ normal
        mask &= dots <= 1e-6  # 可微调容差

    return mask  # shape: (N, ), bool

def filter_grippers_by_box_volume(
    gripper_list,
    T_object_world,
    pcd: o3d.geometry.PointCloud,
    finger_width=15,
    finger_length=100,
    min_points_threshold=1,
):
    points_world = np.asarray(pcd.points)
    retained = []

    for g in gripper_list:
        T_gripper_object = g['T_gripper_object']
        T_gripper_world = T_object_world @ T_gripper_object
        opening = g['opening']

        box_corners_world = get_transformed_box_corners(
            T_gripper_world, opening, finger_width, finger_length
        )
        # visualize_gripper_box_and_pointcloud(box_corners_world, pcd) ##显示夹爪角点

        inside_mask = is_points_in_convex_box(points_world, box_corners_world)
        num_inside = np.sum(inside_mask)

        if num_inside >= min_points_threshold:
            retained.append(g)

    print(f"原始夹爪数量: {len(gripper_list)}")
    print(f"保留夹爪（夹爪包围点 ≥ {min_points_threshold}）数量: {len(retained)}")
    return retained

#########################obb包围盒检测点云
def create_gripper_inner_box(T_gripper_world, opening, finger_length, finger_thickness=15):
    """
    构造夹爪内部空间的 OBB，位于两指之间，Z 方向为抓取方向。
    OBB 已直接在世界坐标系中构造。
    """
    # OBB 尺寸：X (厚度), Y (开口), Z (深度)
    extent = np.array([finger_thickness, opening, finger_length])

    # OBB 中心在夹爪坐标系下的位置
    center_local = np.array([0, 0, -finger_length/2])

    # 变换到世界坐标系
    R = T_gripper_world[:3, :3]
    t = T_gripper_world[:3, 3]
    center_world = R @ center_local + t

    # 构造世界坐标系下的 OBB
    obb_world = o3d.geometry.OrientedBoundingBox(center_world, R, extent)
    return obb_world


def visualize_obb_and_inner_points(pcd, obb, show_box=True, box_color=[0, 1, 0]):
    """
    显示点云，并将落在 OBB 内部的点高亮为红色，其他为灰色，可选显示 OBB 框。

    参数：
        pcd : open3d.geometry.PointCloud
        obb : open3d.geometry.OrientedBoundingBox
        show_box : bool，是否显示 OBB 线框
        box_color : list，OBB 边框颜色 (默认绿色)
    """
    # 复制点云避免原始数据被修改
    pcd_copy = copy.deepcopy(pcd)

    # 获取 OBB 内点的索引
    indices = obb.get_point_indices_within_bounding_box(pcd_copy.points)
    mask = np.zeros(len(pcd_copy.points), dtype=bool)
    mask[indices] = True

    # 设置颜色：红色为在盒子里的点，灰色为其他
    colors = np.tile([0.6, 0.6, 0.6], (len(pcd_copy.points), 1))  # 默认灰色
    colors[mask] = [1.0, 0.0, 0.0]  # 在 OBB 中的为红色
    pcd_copy.colors = o3d.utility.Vector3dVector(colors)

    # 构建 OBB 线框
    geometries = [pcd_copy]
    if show_box:
        obb_points = np.asarray(obb.get_box_points())
        lines = [
            [0, 1], [1, 3], [3, 2], [2, 0],
            [4, 5], [5, 7], [7, 6], [6, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(obb_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([box_color] * len(lines))
        geometries.append(line_set)

    # 显示
    o3d.visualization.draw_geometries(geometries)

def estimate_normals_from_points(points, radius=20, max_nn=30):
    """
    对一组点计算法线，返回 (N, 3) ndarray
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    normals = np.asarray(pcd.normals)
    return normals

def filter_grippers_with_pointcloud_intersection(
    gripper_list,
    T_object_world,
    pcd: o3d.geometry.PointCloud,
    min_points_threshold: int = 1,
    normals_world_all: np.ndarray = None,  # 新增参数
):
    if normals_world_all is None:
        # 如果没传法线，就自己估计
        print("未传入法线，自动估计...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
        )
        normals_world_all = np.asarray(pcd.normals)

    retained = []
    points_world = np.asarray(pcd.points)

    for i, g in enumerate(gripper_list):
        T_gripper_object = g['T_gripper_object']
        T_gripper_world = T_object_world @ T_gripper_object
        T = T_gripper_world
        opening = g['opening']
        finger_length = 100
        finger_width = 15

        obb = create_gripper_inner_box(T, opening, finger_length)
        indices = obb.get_point_indices_within_bounding_box(pcd.points)

        if len(indices) >= min_points_threshold:
            inner_points_world = points_world[indices]
            inner_normals_world = normals_world_all[indices]

            T_world_gripper = np.linalg.inv(T)
            R_inv = T_world_gripper[:3, :3]
            t_inv = T_world_gripper[:3, 3:4]

            points_local = (R_inv @ inner_points_world.T + t_inv).T
            normals_local = (R_inv @ inner_normals_world.T).T

            g_new = g.copy()
            g_new['inner_points_local'] = points_local
            g_new['inner_normals_local'] = normals_local
            g_new['length'] = finger_length
            g_new['width'] = finger_width

            retained.append(g_new)

    print(f"原始夹爪数量: {len(gripper_list)}")
    print(f"保留夹爪（与点云相交 ≥ {min_points_threshold} 点）数量: {len(retained)}")
    return retained

######################可视化夹爪原点
def create_sphere_at(point, radius=2.0, color=[0, 1, 0]):
    """创建一个彩色球体用于显示夹爪原点"""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(point)
    return sphere

def visualize_gripper_origins_from_object_frame(moved_grippers, T_object_world, radius=2.0):
    """
    将 moved_grippers 中的原点（物体坐标系）变换为世界坐标系下绿色球体可视化
    返回：小球 mesh 列表
    """
    spheres = []
    for gripper in moved_grippers:
        origin_obj = gripper['origin']  # 原点在物体坐标系
        origin_world = (T_object_world[:3, :3] @ origin_obj) + T_object_world[:3, 3]
        sphere = create_sphere_at(origin_world, radius=radius, color=[0, 1, 0])
        spheres.append(sphere)
    return spheres

@profiled("detect.grasp_detect.total")
def grasp_detect(ply_path,i):
    #下采样的物体点云，AABB包围框中心坐标，物体坐标系，采样点，采样平面坐标系集合，世界坐标系，采样平面，采样平面坐标系物理模型，物体到世界转换矩阵
    profiler = active_profiler()
    vis_list = []
    vis_list1 = []
    depth_profile = DepthProfiler() if profiler.enabled else None
    opening_profile = OpeningProfiler() if profiler.enabled else None
    profiler.attach_depth_profile(depth_profile)
    profiler.attach_opening_profile(opening_profile)
    cloud_down, obj_center, obj_axes, sample_points, frames, object_world_axis, projections, frame_arrows_list, T_object_world = profiler.measure(
        "detect.frames_process", frames_process, ply_path)

    frame = frames[i - 1]
    # 获取点云坐标（从 pcd 或 downsampled 点云）
    pts = np.asarray(cloud_down.points)  # 或 frames_process 中直接返回
    # frames_process samples on (object_radius + 100 mm); derive the same
    # object-radius convention instead of using the smaller point-cloud max norm.
    sample_radius = float(np.linalg.norm(sample_points[0] - obj_center))
    object_radius = sample_radius - 100.0
    if object_radius <= 0 or not np.isfinite(object_radius):
        object_radius = float(np.max(np.linalg.norm(pts - obj_center, axis=1)))
    depths = generate_depth_samples(object_radius, num_depth=16, max_ratio=1.2)
    profiler.count("depth.object_radius_mm", round(object_radius, 4))
    profiler.count("depth.max_sample_mm", round(float(depths[-1]), 4))
    profiler.count("depth.sample_count", len(depths))
    origin = frame['origin']
    # origin = frame['origin'] + frame['x_axis']*75 - frame['y_axis']*10
    cyl0, cyl1, center0, center1 = profiler.measure(
        "detect.cylinder_sections",
        generate_cylinder_sections,
        origin,
        pcd_points=pts,
        frame=frame,
        cyl_radius=75.0,
        offset=40.0,
        height=1.0,
    )

##########################创建了中间开口为0 的夹爪
    # 创建自定义夹爪
    custom_gripper = create_gripper_model(
        finger_length=100,
        opening=0,
        finger_color=[0.2, 0.5, 0.8]
    )

    # 放置夹爪到第 i 个坐标系
    gripper_meshes_transformed, T_gripper_world, T_gripper_object, axes_matrix, origin = transform_gripper_to_frame(
        gripper_meshes=custom_gripper['meshes'],
        frame=frame,
        origin=center0,
        T_object_world=T_object_world
    )

######################################验证夹爪坐标系转换矩阵是否正确
    gripper_data = create_gripper_model(
        finger_length=100,
        opening=50,
        finger_color=[0.2, 0.2, 0.2]
    )
    gripper_meshes = gripper_data['meshes']
    # 输出：变换后的夹爪
    transformed_gripper = transform_gripper_via_object_frame(
        gripper_meshes, T_gripper_object, T_object_world
    )
    ###,*transformed_gripper
##############################################

######################生成系列夹爪

    gripper_variants = profiler.measure(
        "detect.generate_gripper_variants",
        generate_gripper_variants,
        base_gripper=custom_gripper['model'],
        T_object_world=T_object_world,
        T_gripper_object=T_gripper_object,
        step_deg=15,
        max_deg=179,
        step_open=15,
        max_open=150,
    )
    profiler.count("candidates.gripper_variants", len(gripper_variants))
    # 合并所有夹爪网格
    all_gripper_meshes = []
    for variant in gripper_variants:
        meshes = variant['meshes']
        if isinstance(meshes, list):
            all_gripper_meshes.extend(meshes)
        else:
            all_gripper_meshes.append(meshes)
    # #####单个显示
    # N = 50
    # selected = gripper_variants[N]
    # # + selected['meshes']

#################生成沿Z轴平移的夹爪系列
    # 第二步：每个夹爪沿自身Z轴平移生成新夹爪集合
    moved_gripper_variants = profiler.measure(
        "detect.slide_grippers",
        slide_gripper_along_z,
        gripper_variants=gripper_variants,
        T_object_world=T_object_world,
        step_mm=10,
        max_distance=150,
        depths=depths,
    )
    raw_candidates = moved_gripper_variants
    rule_valid_candidates = filter_structurally_valid_grippers(raw_candidates)
    zero_depth_count = sum(candidate.get("depth", 0) <= 0 for candidate in raw_candidates)
    zero_opening_count = sum(candidate.get("opening", 0) <= 0 for candidate in raw_candidates)
    zero_both_count = sum(
        candidate.get("depth", 0) <= 0 and candidate.get("opening", 0) <= 0
        for candidate in raw_candidates
    )
    profiler.count("candidates.raw", len(raw_candidates))
    profiler.count("candidates.rule_valid", len(rule_valid_candidates))
    profiler.count("rejected.zero_depth", zero_depth_count)
    profiler.count("rejected.zero_opening", zero_opening_count)
    profiler.count("rejected.zero_both", zero_both_count)
    profiler.count("rejected.structural_total", len(raw_candidates) - len(rule_valid_candidates))
    profiler.count("candidates.before_collision", len(rule_valid_candidates))
    def record_candidate_funnel(candidates, phase):
        for candidate in candidates:
            if 'depth' in candidate:
                profiler.group_count("depth", candidate['depth'], phase)
            variant_id = candidate.get('base_id', candidate.get('id'))
            if variant_id is not None:
                profiler.group_count("variant", variant_id, phase)
                if 'depth' in candidate:
                    profiler.matrix_count("variant_depth", variant_id, candidate['depth'], phase)

    record_candidate_funnel(moved_gripper_variants, "candidate")
    record_candidate_funnel(rule_valid_candidates, "rule_valid")
    #合并所有夹爪网络
    all_meshes = []
    for item in moved_gripper_variants:
        all_meshes.extend(item['meshes'])
    #####合并指定深度层所在的夹爪网络
    target_depths = {25, 50, 75}  # 用集合快速判断匹配
    filtered_meshes = []
    for item in moved_gripper_variants:
        if item['depth'] in target_depths:
            filtered_meshes.extend(item['meshes'])

######################碰撞检测
    # 假设点云为 cloud_down，夹爪集合为 moved_gripper_variants
    non_colliding_grippers = profiler.measure(
        "detect.collision_filter",
        filter_collision_free_grippers,
        rule_valid_candidates,
        point_cloud=cloud_down,
        threshold=3.0,
    )
    profiler.count("candidates.collision_free", len(non_colliding_grippers))
    record_candidate_funnel(non_colliding_grippers, "collision_free")
    # 展示无碰撞的所有夹爪
    non_colliding_grippers_mesh_list = []
    for g in non_colliding_grippers:
        non_colliding_grippers_mesh_list.extend(g['meshes'])

######################保留每层开口最小的不同角度的夹爪
    min_opening_grippers = profiler.measure(
        "detect.min_opening_filter",
        filter_by_min_opening_per_depth_angle,
        non_colliding_grippers,
    )
    profiler.count("candidates.after_min_opening", len(min_opening_grippers))
    min_opening_grippers_meshes = []
    for g in min_opening_grippers:
        min_opening_grippers_meshes.extend(g['meshes'])

######################物体点云法线估计
    profiler.measure(
        "detect.final_normal_estimation",
        cloud_down.estimate_normals,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30),
    )
    profiler.measure("detect.final_normal_orientation", cloud_down.orient_normals_consistent_tangent_plane, k=10)
    normals_world_all = np.asarray(cloud_down.normals)

# #######################去掉开口为0 且二指之间无物体点云的夹爪
#     candidate_grippers = filter_grippers_with_object_between_fingers(T_object_world,
#         min_opening_grippers, point_cloud=cloud_down, finger_length=100
#     )
#     candidate_grippers_meshes = []
#     for k in candidate_grippers:
#         candidate_grippers_meshes.extend(k['meshes'])

# ##########################夹爪转移到世界坐标系中
#     filtered = filter_grippers_by_box_volume(
#         gripper_list=min_opening_grippers,
#         T_object_world=T_object_world,
#         pcd=cloud_down,
#         finger_width=15,
#         finger_length=100,
#         min_points_threshold=1,
#         )
#     candidate_grippers_meshes = []
#     for k in filtered:
#         candidate_grippers_meshes.extend(k['meshes'])

##########################OBB包围盒检测内部点云
    filtered = filter_grippers_with_pointcloud_intersection(min_opening_grippers, T_object_world, cloud_down, min_points_threshold=5,normals_world_all=normals_world_all)
    profiler.count("candidates.final_intersection", len(filtered))
    record_candidate_funnel(filtered, "final")
    candidate_grippers_meshes = []
    for k in filtered:
        candidate_grippers_meshes.extend(k['meshes'])

    if depth_profile is not None:
        collision_ids = {candidate['id'] for candidate in non_colliding_grippers}
        opening_ids = {candidate['id'] for candidate in min_opening_grippers}
        final_by_id = {candidate['id']: candidate for candidate in filtered}
        for candidate in moved_gripper_variants:
            candidate_id = candidate['id']
            final_candidate = final_by_id.get(candidate_id)
            contact_points = len(final_candidate.get('inner_points_local', ())) if final_candidate else 0
            depth_profile.add(
                variant_id=candidate['base_id'],
                depth_id=candidate['depth_id'],
                depth_value=candidate['depth'],
                depth_ratio=(candidate['depth'] / object_radius) if object_radius else 0.0,
                collision_free=candidate_id in collision_ids,
                opening_valid=candidate_id in opening_ids,
                intersection_valid=final_candidate is not None,
                contact_points=contact_points,
                surface_point_count=len(pts),
                candidate_id=candidate_id,
            )

    if opening_profile is not None:
        collision_ids = {candidate['id'] for candidate in non_colliding_grippers}
        opening_ids = {candidate['id'] for candidate in min_opening_grippers}
        final_ids = {candidate['id'] for candidate in filtered}
        for candidate in moved_gripper_variants:
            opening_profile.add(
                candidate_id=candidate['id'],
                depth_id=candidate['depth_id'],
                depth_value=candidate['depth'],
                angle_deg=candidate['angle_deg'],
                opening=candidate['opening'],
                structural_valid=(candidate['opening'] > 0 and candidate['depth'] > 0),
                collision_free=candidate['id'] in collision_ids,
                opening_selected=candidate['id'] in opening_ids,
                final_valid=candidate['id'] in final_ids,
            )

#########################可视化过程中夹爪原点
    origin_spheres = visualize_gripper_origins_from_object_frame(moved_gripper_variants, T_object_world, radius=2.0)

    vis_list = []
    # 可视化
    if cyl0 is not None and cyl1 is not None:
        cyl0.paint_uniform_color([1.0, 0.0, 0.0])  # 红
        cyl1.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝
        # 显示第i个
        # vis_list = [cloud_down, cyl0, *object_world_axis] + [projections[i - 1]] + frame_arrows_list[
        #     i - 1] +origin_spheres
        vis_list = [cloud_down, cyl0, *object_world_axis] + [projections[i - 1]] + frame_arrows_list[i - 1]+candidate_grippers_meshes
        vis_list1 = [cloud_down, cyl0, *object_world_axis] + [projections[i - 1]] + frame_arrows_list[
            i - 1]
        # vis_list1 = [cloud_down, cyl0, *object_world_axis] + [projections[i - 1]] + frame_arrows_list[
        #     i - 1] + candidate_grippers_meshes
        # 显示前i个
        # vis_list = [cloud_down, cyl0, cyl1, *object_world_axis] + projections[:i]
        # for arrows in frame_arrows_list[:i]:
        #     vis_list.extend(arrows)

        # o3d.visualization.draw_geometries(vis_list, window_name=f"Cylinder at dir {i}")
        # o3d.visualization.draw_geometries(vis_list1, window_name=f"Cylinder at dir {i}")

    return cloud_down,filtered,candidate_grippers_meshes,vis_list,vis_list1

# if __name__ == "__main__":
#     ply_path = "model/huixing.ply"
#     grasp_detect(ply_path,194)
