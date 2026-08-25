import sys
import math
import open3d as o3d
import numpy as np
import copy
from sklearn.neighbors import KDTree
from profiling import active_profiler, profiled


def estimate_normals(pcd, radius=0.05, max_nn=30):
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )

def build_kdtree(pcd):
    return KDTree(np.asarray(pcd.points))

def ritter_sphere(points):
    # Ritter’s 算法
    pts = points
    p = pts[np.random.randint(len(pts))]
    q = pts[np.argmax(np.linalg.norm(pts - p, axis=1))]
    r = pts[np.argmax(np.linalg.norm(pts - q, axis=1))]
    center = (q + r) / 2.0
    radius = np.linalg.norm(q - r) / 2.0
    for x in pts:
        d = np.linalg.norm(x - center)
        if d > radius:
            new_r = (radius + d) / 2.0
            center += (x - center) / d * (new_r - radius)
            radius = new_r
    return center, radius

def fibonacci_sphere(samples=1):
    pts = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1.0 - (i / float(samples - 1)) * 2.0
        r = math.sqrt(max(0.0, 1.0 - y*y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts.append([x, y, z])
    return np.array(pts)

def build_local_frames(dirs, center, sample_radius, obj_x_axis):
    frames = []
    last_x = None

    for i, dir_vec in enumerate(dirs):
        plane_pt = center + dir_vec * sample_radius
        z_axis = -dir_vec / np.linalg.norm(dir_vec)  # 指向球心，单位化

        # obj_x_axis 在平面上的投影
        proj_x = obj_x_axis - np.dot(obj_x_axis, dir_vec) * dir_vec
        norm_proj_x = np.linalg.norm(proj_x)

        if norm_proj_x < 1e-3:
            if last_x is not None:
                proj_x = last_x - np.dot(last_x, dir_vec) * dir_vec
                proj_x /= np.linalg.norm(proj_x)
            else:
                proj_x = np.cross(z_axis, np.array([1, 0, 0]))
                if np.linalg.norm(proj_x) < 1e-3:
                    proj_x = np.cross(z_axis, np.array([0, 1, 0]))
                proj_x /= np.linalg.norm(proj_x)
        else:
            proj_x /= norm_proj_x

        y_axis = np.cross(z_axis, proj_x)
        y_axis /= np.linalg.norm(y_axis)
        R = np.stack([proj_x, y_axis, z_axis], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = plane_pt
        # 添加 plane id 字段
        frames.append({
            'id': i,                    # 平面编号
            'origin': plane_pt,         # 原点
            'x_axis': proj_x,           # 局部 X
            'y_axis': y_axis,           # 局部 Y
            'z_axis': z_axis,            # 局部 Z
            'transform': T
        })

        last_x = proj_x

    return frames
def create_frame_visuals(origin, x_axis, y_axis, z_axis, length=0.05):
    """
    创建可视化坐标系箭头：X（红）、Y（绿）、Z（蓝）
    """
    arrows = []

    def create_arrow(dir_vec, color):
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.5,
            cone_radius=1,
            cylinder_height=length * 0.8,
            cone_height=length * 0.2
        )
        arrow.paint_uniform_color(color)
        # 旋转箭头（默认沿 Z+）
        default = np.array([0, 0, 1])
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        v = np.cross(default, dir_vec)
        if np.linalg.norm(v) > 1e-6:
            c = np.dot(default, dir_vec)
            s = np.linalg.norm(v)
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
            arrow.rotate(R, center=np.array([0, 0, 0]))
        arrow.translate(origin)
        return arrow

    arrows.append(create_arrow(x_axis, [1, 0, 0]))  # X - 红
    arrows.append(create_arrow(y_axis, [0, 1, 0]))  # Y - 绿
    arrows.append(create_arrow(z_axis, [0, 0, 1]))  # Z - 蓝

    return arrows

def project_to_plane(pts, plane_point, plane_normal):
    n = plane_normal / np.linalg.norm(plane_normal)
    v = pts - plane_point
    d = np.dot(v, n)
    return pts - np.outer(d, n)

@profiled("cloud.frames_process.total")
def frames_process(ply_path,
         voxel_ratio=0.01,
         n_samples=500,
         extra_offset=100):
    # 1. 读点云
    profiler = active_profiler()
    pcd = profiler.measure("cloud.load_point_cloud", o3d.io.read_point_cloud, ply_path)
    ##单位转换
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * 1000)  # m -> mm
    print(f"Loaded: {ply_path}, pts = {len(pcd.points)}")

    # 2. 自动体素下采样
    bounds = pcd.get_max_bound() - pcd.get_min_bound()
    voxel_size = max(bounds) * voxel_ratio
    print(f"Auto voxel_size = {voxel_size:.6f}")
    down = profiler.measure("cloud.voxel_down_sample", pcd.voxel_down_sample, voxel_size)
    print(f"Downsampled pts = {len(down.points)}")

    # 3. 法线 & KD 树
    profiler.measure("cloud.estimate_normals", estimate_normals, down)
    profiler.measure("cloud.build_kdtree", build_kdtree, down)

    # 3.5 建立物体坐标系：基于 OBB 最小包围盒
    # obb = down.get_oriented_bounding_box()
    # obb.color = (0, 1, 0)  # 绿色框体可视化
    # obj_center = obb.center
    # obj_axes = obb.R  # 3x3 矩阵，列向量为 x, y, z 轴方向
    # axis_len = np.linalg.norm(obb.extent) / 4

    # 3.5 建立物体坐标系：基于 AABB 最小包围盒
    aabb = down.get_axis_aligned_bounding_box()
    aabb.color = (0.0, 1.0, 0.0)  # AABB 框体黄色可视化
    obj_center = aabb.get_center()
    obj_axes = np.eye(3)  # 全局坐标系轴向：X, Y, Z
    axis_len = np.linalg.norm(aabb.get_extent()) / 4


    # 创建原始箭头（沿 z+ 方向）
    x_axis = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=1, cone_radius=2,
                                                    cylinder_height=axis_len, cone_height=5)
    y_axis = copy.deepcopy(x_axis)
    z_axis = copy.deepcopy(x_axis)

    # 分别染色
    x_axis.paint_uniform_color([1, 0, 0])  # X - 红
    y_axis.paint_uniform_color([0, 1, 0])  # Y - 绿
    z_axis.paint_uniform_color([0, 0, 1])  # Z - 蓝

    # 旋转箭头方向（默认箭头是沿 z+）
    # 通过旋转将 z+ 转换为 obb 的 x/y/z
    def rotate_arrow_to_direction(arrow, direction):
        direction = direction / np.linalg.norm(direction)
        z = np.array([0, 0, 1])
        v = np.cross(z, direction)
        if np.linalg.norm(v) < 1e-6:
            return  # 不需要旋转
        c = np.dot(z, direction)
        s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        arrow.rotate(R, center=np.array([0, 0, 0]))

    rotate_arrow_to_direction(x_axis, obj_axes[:, 0])
    rotate_arrow_to_direction(y_axis, obj_axes[:, 1])
    rotate_arrow_to_direction(z_axis, obj_axes[:, 2])

    # 平移到物体中心
    x_axis.translate(obj_center)
    y_axis.translate(obj_center)
    z_axis.translate(obj_center)

    # # 4. 最小外接球
    pts = np.asarray(down.points)
    # center, radius = ritter_sphere(pts)
    # print(f"Center: {center}, Radius: {radius:.4f}")
    # 4. 最小外接球：基于 OBB 的 8 个顶点
    aabb = down.get_axis_aligned_bounding_box()
    aabb_pts = np.asarray(aabb.get_box_points())  # 获取8个顶点
    center, radius = ritter_sphere(aabb_pts)  # 计算外接球
    # obb_pts = np.asarray(obb.get_box_points())  # OBB 的 8 个顶点
    # center, radius = ritter_sphere(obb_pts)
    print(f"OBB-based Sphere Center: {center}, Radius: {radius:.4f}")

    # 5. 计算采样半径 = radius + 0.150 m
    sample_radius = radius + extra_offset
    print(f"Sampling on sphere of radius = {sample_radius:.4f}")

    # 6. 球面斐波那契采样
    dirs = fibonacci_sphere(n_samples)
    obj_x_axis = obj_axes[:, 0]  # 假设你已有物体坐标系x轴
    frames = profiler.measure("cloud.build_local_frames", build_local_frames, dirs, center, sample_radius, obj_x_axis)
    # 7. 投影
    projections = []
    sample_points = dirs * sample_radius + center
    for i, dir_vec in enumerate(dirs):
        plane_pt = center + dir_vec * sample_radius
        proj_pts = project_to_plane(pts, plane_pt, dir_vec)

        # 可视化点云投影
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(proj_pts)
        projections.append(p)

    # 8. 可视化：显示下采样云、球心、小球以及前 10 个投影
    down.paint_uniform_color([0.7,0.7,0.7])
    mesh_center = o3d.geometry.TriangleMesh.create_sphere(radius*0.02)
    mesh_center.compute_vertex_normals()
    mesh_center.paint_uniform_color([1.0,0.0,0.0])
    mesh_center.translate(center)
    n = 1
    world_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0])
    object_world_axis = (x_axis, y_axis, z_axis, world_coord)
    vis_list = [down, mesh_center, aabb,
                x_axis, y_axis, z_axis,
                world_coord] + projections[:n]
    # 8.5 可视化前 10 个局部坐标系

    # 构造物体坐标系变换矩阵
    T_object_world = np.eye(4)
    T_object_world[:3, :3] = obj_axes  # 列向量为 x/y/z 轴
    T_object_world[:3, 3] = obj_center  # 物体坐标原点

    frame_arrows_list = []

    for frame in frames[:n_samples]:
        arrows = create_frame_visuals(
            origin=frame['origin'],
            x_axis=frame['x_axis'],
            y_axis=frame['y_axis'],
            z_axis=frame['z_axis'],
            length=15
        )
        frame_arrows_list.append(arrows)
    for frame in frames[:n]:
        arrows = create_frame_visuals(
            origin=frame['origin'],
            x_axis=frame['x_axis'],
            y_axis=frame['y_axis'],
            z_axis=frame['z_axis'],
            length=15  # 可根据实际缩放
        )
        vis_list.extend(arrows)
    vis_list2 = [down, mesh_center, aabb,
                *object_world_axis]
    # frame = frames[200]  # 获取第 n 个 frame
    # arrows = create_frame_visuals(
    #     origin=frame['origin'],
    #     x_axis=frame['x_axis'],
    #     y_axis=frame['y_axis'],
    #     z_axis=frame['z_axis'],
    #     length=15  # 可调节箭头长度
    # )

    vis_list.extend(arrows)
    # o3d.visualization.draw_geometries(vis_list,
    #                                   window_name="Projections at radius+150mm",
    #                                   width=1024, height=768)
    # print(f"Object coordinate frame:")
    # print(f"  Origin (center of OBB): {obj_center}")
    # print(f"  X-axis: {obj_axes[:, 0]}")
    # print(f"  Y-axis: {obj_axes[:, 1]}")
    # print(f"  Z-axis: {obj_axes[:, 2]}")

    return down, obj_center, obj_axes, sample_points, frames, object_world_axis, projections, frame_arrows_list, T_object_world

# if __name__ == "__main__":
#     ply_path = "model/yuantai.ply"
#     cloud_down, obj_center, obj_axes, sample_points, frames, object_world_axis, projections, frame_arrows_list, T_object_world = frames_process(
#         ply_path)
