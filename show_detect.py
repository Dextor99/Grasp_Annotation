import sys
import math
import open3d as o3d
import numpy as np
import copy
from cloud_process import frames_process
from gripper_model import create_gripper_model

def generate_cylinder_sections(pcd_points, frame, cyl_radius=75.0, offset=100.0, height=1.0):
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
    origin = frame['origin']
    # origin = frame['origin'] + frame['x_axis'] * 30.0 + frame['y_axis'] * 30.0
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


def transform_gripper_to_frame(gripper_meshes, frame, origin, T_object_world=None):
    """
    将夹爪从局部原始坐标，变换到指定 frame 坐标系，并可计算其在物体坐标系下的姿态

    参数:
        gripper_meshes: 原始 gripper TriangleMesh 列表
        frame: 包含 x/y/z_axis 的字典
        origin: 夹爪在世界坐标系下放置的原点
        T_object_world: 物体 → 世界 坐标系变换 (4x4)，如果提供将返回夹爪在物体坐标系下的变换

    返回:
        transformed_meshes: 已变换的 gripper 模型
        T_world_gripper: 夹爪在世界坐标系下的 4x4 变换矩阵
        T_object_gripper: 夹爪在物体坐标系下的 4x4 变换矩阵（如果传入 T_object_world）
        axes_matrix: gripper 的三个方向向量列向量组成的矩阵
        origin: 原点坐标
    """
    x_axis = frame['x_axis']
    y_axis = frame['y_axis']
    z_axis = frame['z_axis']
    axes_matrix = np.column_stack((x_axis, y_axis, z_axis))  # shape (3, 3)

    T_world_gripper = np.eye(4)
    T_world_gripper[:3, :3] = axes_matrix
    T_world_gripper[:3, 3] = origin

    transformed_meshes = []
    for mesh in gripper_meshes:
        mesh_copy = copy.deepcopy(mesh)
        mesh_copy.transform(T_world_gripper)
        transformed_meshes.append(mesh_copy)

    if T_object_world is not None:
        T_world_object = np.linalg.inv(T_object_world)
        T_object_gripper = T_world_object @ T_world_gripper
    else:
        T_object_gripper = None

    return transformed_meshes, T_world_gripper, T_object_gripper, axes_matrix, origin

#验证夹爪坐标系转换矩阵
def transform_gripper_via_object_frame(gripper_meshes,
                                       T_object_gripper,
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
    T_gripper_world = T_object_world @ T_object_gripper


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
        T_object_gripper: np.ndarray,
        step_deg=5,
        max_deg=180,
        step_open=5,
        max_open=150
):
    T_ref = T_object_world @ T_object_gripper
    variants = []
    idx = 0

    # 提取夹爪的当前旋转和平移
    R_gripper = T_ref[:3, :3]
    t_gripper = T_ref[:3, 3]

    for angle_deg in range(0, max_deg + 1, step_deg):
        theta_rad = np.deg2rad(angle_deg)

        # 绕夹爪局部 Z 轴的旋转矩阵
        Rz = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad), 0],
            [np.sin(theta_rad), np.cos(theta_rad), 0],
            [0, 0, 1]
        ])

        # 关键修改：在夹爪局部坐标系下旋转
        R_new = R_gripper @ Rz  # 左乘 Rz，表示绕局部 Z 轴旋转

        # 构建新的变换矩阵（平移不变）
        T_object_gripper_i = np.eye(4)
        T_object_gripper_i[:3, :3] = R_new
        T_object_gripper_i[:3, 3] = t_gripper

        for opening in range(0, max_open + 1, step_open):
            gripper = copy.deepcopy(base_gripper)
            gripper.opening = opening
            gripper.transform(T_object_gripper_i)
            meshes = gripper.get_meshes()

            variants.append({
                'id': idx,
                'angle_deg': angle_deg,
                'opening': opening,
                'T_object_gripper': T_object_gripper_i.copy(),
                'meshes': meshes,
                'origin': t_gripper.copy(),  # 原点不变
                'model': gripper,
            })
            idx += 1

    return variants

###############生成沿Z轴平移的夹爪集合
def slide_gripper_along_z(
    gripper_variants,
    T_object_world,
    step_mm=4,
    max_distance=100
):
    """
    沿夹爪自身Z轴正方向平移生成新夹爪（先按角度/开口排列，再按深度移动）。

    返回：
        moved_grippers: list，每个元素包含原标签 + 新增的 "depth" 和更新后的姿态
    """
    moved_grippers = []
    idx = 0

    for d in range(0, max_distance + 1, step_mm):  # ✅ depth 在外层
        for variant in gripper_variants:           # ✅ variant 在内层
            base_pose = variant['T_object_gripper']
            base_model = variant['model'] if 'model' in variant else None
            opening = variant['opening']
            angle_deg = variant['angle_deg']
            origin = variant['origin']

            # 局部坐标系下沿 z 平移 d mm
            T_translation = np.eye(4)
            T_translation[:3, 3] = np.array([0, 0, d])
            T_object_gripper_shifted = base_pose @ T_translation

            gripper = copy.deepcopy(base_model)
            gripper.opening = opening
            gripper.transform(T_object_gripper_shifted)

            moved_grippers.append({
                'id': idx,
                'base_id': variant['id'],
                'angle_deg': angle_deg,
                'opening': opening,
                'depth': d,
                'T_object_gripper': T_object_gripper_shifted.copy(),
                'meshes': gripper.get_meshes(),
                'origin': T_object_gripper_shifted[:3, 3],
                'model': gripper,
            })

            idx += 1

    return moved_grippers

#####################碰撞检测
def check_collision(gripper_meshes, point_cloud, threshold=2.0):
    """
    检测夹爪与点云是否碰撞
    参数:
        gripper_meshes: 夹爪网格列表 [finger1, finger2, base, axis]
        point_cloud: 目标点云
        threshold: 碰撞判定阈值(mm)
    返回:
        bool: True表示有碰撞，False表示无碰撞
    """
    # 合并夹爪所有网格
    combined_mesh = gripper_meshes[0] + gripper_meshes[1] + gripper_meshes[2]

    # 创建夹爪的KD树
    gripper_pcd = combined_mesh.sample_points_uniformly(number_of_points=1000)
    gripper_tree = o3d.geometry.KDTreeFlann(gripper_pcd)

    # 检查点云中的每个点
    points = np.asarray(point_cloud.points)
    for pt in points:
        # 查找夹爪中最近的点
        [k, idx, _] = gripper_tree.search_knn_vector_3d(pt, 1)
        nearest_pt = np.asarray(gripper_pcd.points)[idx[0]]
        distance = np.linalg.norm(pt - nearest_pt)

        if distance < threshold:
            return True  # 有碰撞

    return False  # 无碰撞

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
    filtered_list = []
    total = len(gripper_list)
    for i, gripper in enumerate(gripper_list):
        if not check_collision(gripper['meshes'], point_cloud, threshold):
            filtered_list.append(gripper)
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"检测进度：{i+1}/{total} 个夹爪")

    print(f"原始夹爪数量：{total}")
    print(f"无碰撞夹爪数量：{len(filtered_list)}")
    return filtered_list


##########################保留每个depth和旋转角度下，开口最小的一个夹爪
def filter_by_min_opening_per_depth_angle(gripper_list):
    """
    在每个 depth 和 angle_deg 下保留 opening 最小的夹爪。

    参数：
        gripper_list: list，每个元素是 dict，包含 'depth'、'angle_deg'、'opening' 等字段

    返回：
        filtered: list，只保留最小 opening 的夹爪
    """
    from collections import defaultdict

    grouped = defaultdict(list)

    # 分组
    for g in gripper_list:
        key = (g['depth'], g['angle_deg'])
        grouped[key].append(g)

    filtered = []

    # 每组中找 opening 最小的
    for key, group in grouped.items():
        min_gripper = min(group, key=lambda x: x['opening'])
        filtered.append(min_gripper)

    print(f"原始夹爪数：{len(gripper_list)}")
    print(f"每层每角度最优夹爪数：{len(filtered)}")
    return filtered

########################去掉开口为0 且二指之间无物体点云的夹爪
def is_point_between_fingers(point, T_world_gripper, opening, finger_length, finger_thickness=10):
    """
    检查某点是否落在两指之间的包围区域（简化为盒子）。
    夹爪坐标系假设：
        - X 轴：从左指指心 -> 右指指心
        - Z 轴：指向物体，平移方向
        - Y 轴：指尖方向
    """

    # 转换点到夹爪坐标系
    T_gripper_world = np.linalg.inv(T_world_gripper)
    p_local = T_gripper_world[:3, :3] @ point + T_gripper_world[:3, 3]  # R @ p + t

    x, y, z = p_local

    # 假设指间宽度为 opening，中间夹持区域 ±thickness/2，前后为 finger_length
    in_x = (-opening / 2 + 2.0 <= x <= opening / 2 - 2.0)
    in_y = (-finger_thickness / 2 <= y <= finger_thickness / 2)
    in_z = (0 <= z <= finger_length)

    return in_x and in_y and in_z


def filter_grippers_with_object_between_fingers(T_object_world,grippers, point_cloud, finger_length=100):
    """
    过滤开口不为0 且夹爪之间有物体点云的夹爪。

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
        if opening == 0:
            continue
        depth = g['depth']
        if depth == 0:
            continue

        T_object_gripper = g['T_object_gripper']
        T_world_gripper = T_object_world @ T_object_gripper

        found = False
        for pt in points:
            if is_point_between_fingers(pt, T_world_gripper, opening, finger_length):
                found = True
                break

        if found:
            filtered.append(g)

    print(f"原始数量: {len(grippers)}")
    print(f"保留夹爪（开口非0 + 有点云）数量: {len(filtered)}")
    return filtered

def main(ply_path):
    #下采样的物体点云，AABB包围框中心坐标，物体坐标系，采样点，采样平面坐标系集合，世界坐标系，采样平面，采样平面坐标系物理模型，物体到世界转换矩阵
    cloud_down, obj_center, obj_axes, sample_points, frames, object_world_axis, projections, frame_arrows_list, T_object_world = frames_process(
        ply_path)

    i = 194
    frame = frames[i - 1]
    # 获取点云坐标（从 pcd 或 downsampled 点云）
    pts = np.asarray(cloud_down.points)  # 或 frames_process 中直接返回

    cyl0, cyl1, center0, center1 = generate_cylinder_sections(
        pcd_points=pts,
        frame=frame,
        cyl_radius=75.0,  # 直径 150mm
        offset=150.0,  # 第二截面延申 100mm
        height=1.0  # 扁平圆盘
    )

##########################创建了中间开口为0 的夹爪
    # 创建自定义夹爪
    custom_gripper = create_gripper_model(
        finger_length=100,
        opening=0,
        finger_color=[0.2, 0.5, 0.8]
    )

    # 放置夹爪到第 i 个坐标系
    gripper_meshes_transformed, T_world_gripper, T_object_gripper, axes_matrix, origin = transform_gripper_to_frame(
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
        gripper_meshes, T_object_gripper, T_object_world
    )
    ###,*transformed_gripper
##############################################
#
# ######################生成系列夹爪
#     gripper_variants = generate_gripper_variants(
#         base_gripper=custom_gripper['model'],
#         T_object_world = T_object_world,
#         T_object_gripper = T_object_gripper,
#         step_deg=180,
#         max_deg=180,
#         step_open=150,
#         max_open=150
#     )
#     # 合并所有夹爪网格
#     all_gripper_meshes = []
#     for variant in gripper_variants:
#         meshes = variant['meshes']
#         if isinstance(meshes, list):
#             all_gripper_meshes.extend(meshes)
#         else:
#             all_gripper_meshes.append(meshes)
#     # #####单个显示
#     # N = 50
#     # selected = gripper_variants[N]
#     # # + selected['meshes']
#
# #################生成沿Z轴平移的夹爪系列
#     # 第二步：每个夹爪沿自身Z轴平移生成新夹爪集合
#     moved_gripper_variants = slide_gripper_along_z(
#         gripper_variants=gripper_variants,
#         T_object_world=T_object_world,
#         step_mm=25,
#         max_distance=150
#     )
#     #合并所有夹爪网络
#     all_meshes = []
#     for item in moved_gripper_variants:
#         all_meshes.extend(item['meshes'])
#     #####合并指定深度层所在的夹爪网络
#     target_depths = {25, 50, 75}  # 用集合快速判断匹配
#     filtered_meshes = []
#     for item in moved_gripper_variants:
#         if item['depth'] in target_depths:
#             filtered_meshes.extend(item['meshes'])
#
# ######################碰撞检测
#     # 假设点云为 cloud_down，夹爪集合为 moved_gripper_variants
#     non_colliding_grippers = filter_collision_free_grippers(
#         moved_gripper_variants,
#         point_cloud=cloud_down,
#         threshold=2.0  # 单位：mm
#     )
#     # 展示无碰撞的所有夹爪
#     non_colliding_grippers_mesh_list = []
#     for g in non_colliding_grippers:
#         non_colliding_grippers_mesh_list.extend(g['meshes'])
#
# ######################保留每层开口最小的不同角度的夹爪
#     min_opening_grippers = filter_by_min_opening_per_depth_angle(non_colliding_grippers)
#     min_opening_grippers_meshes = []
#     for g in min_opening_grippers:
#         min_opening_grippers_meshes.extend(g['meshes'])
#
# #######################去掉开口为0 且二指之间无物体点云的夹爪
#     candidate_grippers = filter_grippers_with_object_between_fingers(T_object_world,
#         min_opening_grippers, point_cloud=cloud_down, finger_length=100
#     )
#     candidate_grippers_meshes = []
#     for k in candidate_grippers:
#         candidate_grippers_meshes.extend(k['meshes'])

    # 可视化
    if cyl0 is not None and cyl1 is not None:
        cyl0.paint_uniform_color([1.0, 0.0, 0.0])  # 红
        cyl1.paint_uniform_color([0.0, 0.0, 1.0])  # 蓝
        # 显示第i个
        vis_list = [cloud_down, cyl0, *object_world_axis,*transformed_gripper] +gripper_meshes_transformed
        # 显示前i个
        # vis_list = [cloud_down, cyl0, cyl1, *object_world_axis] + projections[:i]
        # for arrows in frame_arrows_list[:i]:
        #     vis_list.extend(arrows)
        o3d.visualization.draw_geometries(vis_list, window_name=f"Cylinder at dir {i}")

if __name__ == "__main__":
    ply_path = "model/huixing.ply"
    main(ply_path)