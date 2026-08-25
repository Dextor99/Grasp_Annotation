import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from grasp_detect import grasp_detect
from numpy.linalg import norm
import csv
from datetime import datetime
from profiling import profiled

@profiled("score_v3.compute_grasp_scores")
def compute_grasp_scores_simple(candidate_grippers, pcd,vis=False):
    """
    计算夹爪抓取评分的前8个指标：
    1. 夹爪内点云数占总点云比例
    2. 内部点云z值最小值（夹爪坐标系）
    3. 内部点云z值最大值（夹爪坐标系）
    4. 内部点云在夹爪平面(xoz)投影凸包面积
    5. 凸包面积与夹爪指面面积比
    6. y轴最小值点和最大值点之间的差值（夹爪坐标系y轴距离）
    7. 内部点极值点法向量与指面夹角（分别与 y+ 和 y- 方向）
    8. 上述两个夹角差的绝对值

    参数：
        candidate_grippers: list[dict]，每个dict包含：
            - 'inner_points_local': (N,3) ndarray，夹爪内点云坐标（夹爪坐标系）
            - 'inner_normals_local': (N,3) ndarray，对应点的法线（夹爪坐标系）
            - 'finger_width': float，夹爪指宽（用于计算指面面积）
            - 'opening': float，夹爪开口（用于计算指面面积）
            - ...其他字段
        pcd: open3d.geometry.PointCloud，物体点云（世界坐标系）

    返回：
        candidate_grippers，原列表，每个dict新增评分字段
    """
    total_points_num = len(pcd.points)

    for g in candidate_grippers:
        inner_pts = g.get('inner_points_local', np.empty((0, 3)))
        inner_normals = g.get('inner_normals_local', np.empty((0, 3)))
        n_inner = inner_pts.shape[0]

        # 1. 内部点数占比
        ratio_inner = n_inner / total_points_num if total_points_num > 0 else 0

        if n_inner == 0:
            g['score_inner_points_ratio'] = ratio_inner
            g['score_zmin'] = None
            g['score_zmax'] = None
            g['score_proj_area'] = 0
            g['score_proj_area_ratio'] = 0
            g['score_y_diff'] = None
            g['score_y0_diff'] = None
            g['score_angle_ymin'] = None
            g['score_angle_ymax'] = None
            g['score_angle_diff'] = None
            continue

        # 2. z最小值
        z_min = np.min(inner_pts[:, 2])
        # 3. z最大值
        z_max = np.max(inner_pts[:, 2])

        # 4. 投影到xoz平面，计算凸包面积
        proj_pts = inner_pts[:, [0, 2]]
        if len(proj_pts) >= 3:
            try:
                hull = ConvexHull(proj_pts)
                proj_area = hull.volume  # 对于2D凸包，volume等于面积
            except:
                proj_area = 0
        else:
            proj_area = 0

        opening = g.get('opening', 50)
        width = g.get('width', 15)
        length = g.get('length', 100)
        finger_area = width * length
        proj_area_ratio = proj_area / finger_area if finger_area > 0 else 0

        y_vals = inner_pts[:, 1]
        y_diff = np.max(y_vals) - np.min(y_vals)
        y0_diff = abs((opening / 2 - np.max(y_vals)) - (np.min(y_vals) + opening / 2))

        # 7. 法向量夹角（极值点）
        score_angle_ymin = None
        score_angle_ymax = None
        score_angle_diff = None

        if inner_normals.shape[0] == n_inner:
            idx_ymin = np.argmin(inner_pts[:, 1])
            idx_ymax = np.argmax(inner_pts[:, 1])
            normal_min = inner_normals[idx_ymin]
            normal_max = inner_normals[idx_ymax]

            y_pos = np.array([0, -1, 0])  # 对应右指内表面
            y_neg = np.array([0, 1, 0])   # 对应左指内表面

            def angle_between(v1, v2):
                v1 = v1 / norm(v1)
                v2 = v2 / norm(v2)
                return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) * 180 / np.pi

            score_angle_ymin = angle_between(normal_min, y_neg)
            score_angle_ymax = angle_between(normal_max, y_pos)
            score_angle_diff = abs(score_angle_ymax - score_angle_ymin)

        force_closure_score = None
        if n_inner >= 2 and inner_normals.shape[0] == n_inner:
            idx_ymin = np.argmin(inner_pts[:, 1])
            idx_ymax = np.argmax(inner_pts[:, 1])
            normal_min = inner_normals[idx_ymin]
            normal_max = inner_normals[idx_ymax]

            # 夹爪Y轴开闭方向：y轴正方向为右指内表面，负方向为左指内表面
            y_pos = np.array([0, -1, 0])  # 右指内表面方向
            y_neg = np.array([0, 1, 0])  # 左指内表面方向

            # 归一化法线
            normal_min /= norm(normal_min) + 1e-8
            normal_max /= norm(normal_max) + 1e-8

            # 计算法线与开闭方向的cos夹角
            cos_min = np.dot(normal_min, y_pos)
            cos_max = np.dot(normal_max, y_neg)

            # 取两者中较小的cos值作为力闭合评分（评分范围[-1,1]，越接近1越好）
            force_closure_score = min(cos_min, cos_max)
        # 赋值
        g['score_inner_points_ratio'] = ratio_inner
        g['score_zmin'] = z_min
        g['score_zmax'] = z_max
        g['score_proj_area'] = proj_area
        g['score_proj_area_ratio'] = proj_area_ratio
        g['score_y_diff'] = y_diff
        g['score_y0_diff'] = y0_diff
        g['score_angle_ymin'] = score_angle_ymin
        g['score_angle_ymax'] = score_angle_ymax
        g['score_angle_diff'] = score_angle_diff
        g['score_force_closure'] = force_closure_score

        # ✅ 可视化夹爪内部点 + 法线 + 极值点
        if vis:
            pcd_inner = o3d.geometry.PointCloud()
            pcd_inner.points = o3d.utility.Vector3dVector(inner_pts)
            if inner_normals.shape[0] == inner_pts.shape[0]:
                pcd_inner.normals = o3d.utility.Vector3dVector(inner_normals)
            pcd_inner.paint_uniform_color([0.2, 0.8, 0.2])  # 绿色

            spheres = []
            if n_inner >= 2:
                sp1 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                sp1.translate(inner_pts[idx_ymin])
                sp1.paint_uniform_color([1.0, 0.0, 0.0])  # 红色（y最小）

                sp2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
                sp2.translate(inner_pts[idx_ymax])
                sp2.paint_uniform_color([1.0, 1.0, 0.0])  # 黄色（y最大）

                spheres = [sp1, sp2]

            o3d.visualization.draw_geometries(
                [pcd_inner] + spheres,
                point_show_normal=True,
                window_name='Gripper Internal Points & Normals'
            )

    return candidate_grippers


def visualize_grippers_with_pointcloud_and_normals(pcd, gripper_meshes):
    """
    同时显示夹爪模型（多个）、物体点云、法线。

    参数：
        pcd: open3d.geometry.PointCloud，物体点云
        gripper_meshes: list of open3d.geometry.TriangleMesh，夹爪网格
    """
    # 估计法线
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=10)

    # 可视化列表：夹爪 + 点云
    vis_list = [pcd] + gripper_meshes

    o3d.visualization.draw_geometries(
        vis_list,
        point_show_normal=True,  # 显示点云法线
        mesh_show_back_face=True,  # 可选：显示网格背面
        window_name="Grippers + PointCloud + Normals"
    )

def visualize_grippers_inner_points_normals(
    pcd,
    candidate_grippers,
    point_size=5,
    normal_length=5,
    sphere_radius=1,
):
    """
    可视化：
    - 物体点云（不显示法线）
    - 所有夹爪模型
    - 每个夹爪内部点（红色）
    - 每个夹爪内部点的法线（绿色）
    - 每个夹爪内部 y 极值点（蓝球、黄球）

    参数：
        pcd: open3d.geometry.PointCloud，物体点云
        candidate_grippers: list[dict]，每个 dict 包含：
            - 'mesh': TriangleMesh
            - 'inner_points_local': (N,3)
            - 'inner_normals_local': (N,3)
            - 'T_object_world': (4,4)
            - 'T_gripper_object': (4,4)
    """
    vis_geometries = []

    # 添加点云（灰色）
    pcd_down = pcd.voxel_down_sample(voxel_size=0.001)
    pcd_down.paint_uniform_color([0.7, 0.7, 0.7])
    vis_geometries.append(pcd_down)

    for g in candidate_grippers:
        # 提取信息
        meshs = g.get('meshes')
        pts_local = g.get('inner_points_local')
        normals_local = g.get('inner_normals_local')
        T_object_world = g.get('T_object_world')
        T_gripper_object = g.get('T_gripper_object')

        # 确保信息完整
        if meshs is None or pts_local is None or normals_local is None or T_object_world is None or T_gripper_object is None:
            continue
        # 展开多个 mesh 加入 vis 列表
        vis_geometries.extend(meshs)

        # 夹爪局部 -> 世界坐标系
        T = T_object_world @ T_gripper_object
        R = T[:3, :3]
        t = T[:3, 3:4]

        pts_world = (R @ pts_local.T + t).T
        normals_world = (R @ normals_local.T).T

        # 内部点可视化（红色）
        pcd_inner = o3d.geometry.PointCloud()
        pcd_inner.points = o3d.utility.Vector3dVector(pts_world)
        pcd_inner.paint_uniform_color([1, 0, 0])
        vis_geometries.append(pcd_inner)

        # 法线（绿色线段）
        lines = []
        line_pts = []
        for i, (p, n) in enumerate(zip(pts_world, normals_world)):
            line_pts.append(p)
            line_pts.append(p + n * normal_length)
            lines.append([2 * i, 2 * i + 1])
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(line_pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector([[0, 1, 0]] * len(lines))
        vis_geometries.append(line_set)

        # y轴极值点（蓝黄球）
        y_vals = pts_local[:, 1]
        idx_ymin = np.argmin(y_vals)
        idx_ymax = np.argmax(y_vals)

        sphere_min = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_min.paint_uniform_color([0, 0, 1])  # 蓝色：y最小
        sphere_min.translate(pts_world[idx_ymin])
        vis_geometries.append(sphere_min)

        sphere_max = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere_max.paint_uniform_color([1, 1, 0])  # 黄色：y最大
        sphere_max.translate(pts_world[idx_ymax])
        vis_geometries.append(sphere_max)

    # 显示
    o3d.visualization.draw_geometries(vis_geometries, window_name="Grippers + Inner Points + Normals")


def save_grasp_scores_to_csv(candidate_grippers, output_dir='./score_record'):
    """
    将 candidate_grippers 中所有评分与参数保存为 CSV 文件，文件名中包含数量
    例如: grasp_scores_6.csv
    """
    if not candidate_grippers:
        print("❌ candidate_grippers 为空，未保存文件")
        return

    num_grasps = len(candidate_grippers)
    # output_path = f"{output_dir.rstrip('/')}/grasp_scores_{num_grasps}.csv"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"{output_dir.rstrip('/')}/grasp_scores_{num_grasps}_{timestamp}.csv"

    # 获取所有字段
    all_keys = set()
    for g in candidate_grippers:
        all_keys.update(g.keys())
    all_keys = sorted(all_keys)

    # 排除不能写入CSV的字段
    excluded_keys = set()
    for key in all_keys:
        for g in candidate_grippers:
            val = g.get(key, None)
            if isinstance(val, (dict, list, tuple, set, np.ndarray)) or hasattr(val, 'geometry_type'):
                excluded_keys.add(key)
                break
    valid_keys = [k for k in all_keys if k not in excluded_keys]

    # 写入CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=valid_keys)
        writer.writeheader()
        for g in candidate_grippers:
            row = {k: g.get(k, None) for k in valid_keys}
            writer.writerow(row)

    print(f"✅ 成功保存抓取评分到文件: {output_path}")

if __name__ == "__main__":
    ply_path = "model/huixing.ply"
    i=194
    cloud_down,candidate_grippers,candidate_grippers_meshes,vis_list = grasp_detect(ply_path,i)

    candidate_grippers = compute_grasp_scores_simple(candidate_grippers, cloud_down, vis=False)
    save_grasp_scores_to_csv(candidate_grippers)

    #显示物体点云所有点法线
    # visualize_grippers_with_pointcloud_and_normals(cloud_down, candidate_grippers_meshes)
    # 显示夹爪内点云法线
    visualize_grippers_inner_points_normals(cloud_down, candidate_grippers)


    # o3d.visualization.draw_geometries(vis_list, window_name=f"Cylinder at dir {i}")
