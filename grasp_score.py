import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
from grasp_detect import grasp_detect


def compute_grasp_scores_simple(candidate_grippers, pcd):
    """
    计算夹爪抓取评分的前6个指标（不含法线相关）：
    1. 夹爪内点云数占总点云比例
    2. 内部点云z值最小值（夹爪坐标系）
    3. 内部点云z值最大值（夹爪坐标系）
    4. 内部点云在夹爪平面(xoz)投影凸包面积
    5. 凸包面积与夹爪指面面积比
    6. y轴最小值点和最大值点之间的差值（夹爪坐标系y轴距离）

    参数：
        candidate_grippers: list[dict]，每个dict包含
            - 'inner_points_local': (N,3) ndarray，夹爪内点云坐标（夹爪坐标系）
            - 'finger_width': float，夹爪指宽（用于计算指面面积）
            - 'opening': float，夹爪开口（用于计算指面面积）
            - ...其他字段
        pcd: open3d.geometry.PointCloud，物体点云（世界坐标系）
    返回：
        candidate_grippers，原列表，每个dict新增评分字段
    """
    total_points_num = len(pcd.points)

    for g in candidate_grippers:
        inner_pts = g.get('inner_points_local', np.empty((0,3)))
        n_inner = inner_pts.shape[0]

        # 1. 内部点数占比
        ratio_inner = n_inner / total_points_num if total_points_num > 0 else 0

        if n_inner == 0:
            # 点数为0，后面指标无意义
            g['score_inner_points_ratio'] = ratio_inner
            g['score_zmin'] = None
            g['score_zmax'] = None
            g['score_proj_area'] = 0
            g['score_proj_area_ratio'] = 0
            g['score_y_diff'] = None
            continue

        # 2. z最小值
        z_min = np.min(inner_pts[:,2])
        # 3. z最大值
        z_max = np.max(inner_pts[:,2])

        # 4. 投影到xoz平面，计算凸包面积
        proj_pts = inner_pts[:, [0,2]]  # x,z
        if len(proj_pts) >= 3:
            try:
                hull = ConvexHull(proj_pts)
                proj_area = hull.volume
            except:
                proj_area = 0
        else:
            proj_area = 0

        opening = g.get('opening', 50)
        # 5. 凸包面积 / 指面面积
        width = g.get('width', 15)  # 默认15
        length = g.get('length', 100)  # 默认100
        finger_area = width * length
        proj_area_ratio = proj_area / finger_area if finger_area > 0 else 0

        # 6. y最大值和y最小值点的差值
        y_vals = inner_pts[:,1]
        y_diff = np.max(y_vals) - np.min(y_vals)
        y0_diff = abs((opening / 2 - np.max(y_vals)) - (np.min(y_vals) + opening / 2))
        # 赋值
        g['score_inner_points_ratio'] = ratio_inner
        g['score_zmin'] = z_min
        g['score_zmax'] = z_max
        g['score_proj_area'] = proj_area
        g['score_proj_area_ratio'] = proj_area_ratio
        g['score_y_diff'] = y_diff
        g['score_y0_diff'] = y0_diff

    return candidate_grippers

if __name__ == "__main__":
    ply_path = "model/huixing.ply"
    i=0
    cloud_down,candidate_grippers,candidate_grippers_meshes,vis_list = grasp_detect(ply_path,i)

    candidate_grippers = compute_grasp_scores_simple(candidate_grippers, cloud_down)

    o3d.visualization.draw_geometries(vis_list, window_name=f"Cylinder at dir {i}")