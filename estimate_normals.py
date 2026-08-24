import open3d as o3d

def estimate_and_visualize_normals(pcd_path, radius=10, max_nn=30):
    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    print(f"读取点云，共有点数: {len(pcd.points)}")

    # 2. 估计法线（局部邻域）
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius,  # 邻域半径
        max_nn=max_nn   # 邻域最大点数
    ))

    # 3. 可选：方向一致化（可选操作）
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # 4. 打印部分法线信息（前 5 个）
    normals = pcd.normals
    for i in range(min(5, len(normals))):
        print(f"Point {i} normal: {normals[i]}")

    # 5. 显示点云和法线
    o3d.visualization.draw_geometries([pcd],
        point_show_normal=True,
        window_name="点云与法线",
        width=800,
        height=600
    )

    return pcd

# 示例使用
if __name__ == "__main__":
    path = "model/huixing.ply"  # 修改为你自己的文件路径
    pcd_with_normals = estimate_and_visualize_normals(path)
