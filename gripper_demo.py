import open3d as o3d
import numpy as np
import copy
from gripper_model import GripperModel

def gripper_demo():
    # ——— 夹爪参数设置（保留） ———
    finger_thickness = 5.0    # mm
    finger_width     = 10.0   # mm
    finger_length    = 100.0  # mm
    base_depth       = 20.0   # mm
    base_color       = [0.2, 0.8, 0.2]
    finger_color     = [0.8, 0.2, 0.2]
    axis_size        = 10.0   # mm

    # 开口与旋转步长
    min_open  = 0    # 最小开口 mm
    max_open  = 150  # 最大开口 mm
    step_open = 5    # 开口步长 mm
    step_deg  = 5    # 旋转步长 °

    all_meshes = []

    # 对每一个开口值，生成一圈旋转
    for opening in range(min_open, max_open + 1, step_open):
        # 实例化当前开口的夹爪
        gripper = GripperModel(
            finger_thickness=finger_thickness,
            finger_width=finger_width,
            finger_length=finger_length,
            opening=opening,
            base_depth=base_depth,
            base_color=base_color,
            finger_color=finger_color,
            axis_size=axis_size
        )
        base_meshes = gripper.get_meshes()  # [finger1, finger2, base, axis]

        # 绕 Z 轴每隔 step_deg 复制并旋转
        for deg in range(0, 180, step_deg):
            angle_rad = np.deg2rad(deg)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
            for mesh in base_meshes:
                m = copy.deepcopy(mesh)
                m.rotate(R, center=(0, 0, 0))
                all_meshes.append(m)

    # 一次性渲染所有生成的夹爪
    o3d.visualization.draw_geometries(
        all_meshes,
        window_name='Opening & Rotation Sweep',
        width=1280,
        height=800
    )

def gripper_demo1(show_all=True, show_index=None):
    # ——— 夹爪参数设置（保留） ———
    finger_thickness = 5.0    # mm
    finger_width     = 10.0   # mm
    finger_length    = 100.0  # mm
    base_depth       = 20.0   # mm
    base_color       = [0.2, 0.8, 0.2]
    finger_color     = [0.8, 0.2, 0.2]
    axis_size        = 10.0   # mm

    # 开口与旋转步长
    min_open  = 0    # 最小开口 mm
    max_open  = 110  # 最大开口 mm
    step_open = 5    # 开口步长 mm
    step_deg  = 5    # 旋转步长 °

    gripper_dict = {}
    index = 0

    for opening in range(min_open, max_open + 1, step_open):
        gripper = GripperModel(
            finger_thickness=finger_thickness,
            finger_width=finger_width,
            finger_length=finger_length,
            opening=opening,
            base_depth=base_depth,
            base_color=base_color,
            finger_color=finger_color,
            axis_size=axis_size
        )
        base_meshes = gripper.get_meshes()

        for deg in range(0, 180, step_deg):
            angle_rad = np.deg2rad(deg)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])

            transformed_meshes = []
            for mesh in base_meshes:
                m = copy.deepcopy(mesh)
                m.rotate(R, center=(0, 0, 0))
                transformed_meshes.append(m)

            gripper_dict[index] = {
                'opening': opening,
                'rotation_deg': deg,
                'meshes': transformed_meshes
            }
            index += 1

    # 可视化控制
    if show_all:
        all_meshes = []
        for data in gripper_dict.values():
            all_meshes.extend(data['meshes'])
        o3d.visualization.draw_geometries(
            all_meshes,
            window_name='All Grippers',
            width=1280,
            height=800
        )
    elif show_index is not None and show_index in gripper_dict:
        o3d.visualization.draw_geometries(
            gripper_dict[show_index]['meshes'],
            window_name=f'Gripper #{show_index}',
            width=800,
            height=600
        )
    else:
        print("No visualization shown. Set show_all=True or specify show_index.")

    return gripper_dict

def gripper_demo_with_index():
    # 参数设置（与原始代码相同）
    finger_thickness = 5.0
    finger_width = 10.0
    finger_length = 100.0
    base_depth = 20.0
    base_color = [0.2, 0.8, 0.2]
    finger_color = [0.8, 0.2, 0.2]
    axis_size = 10.0

    min_open = 0
    max_open = 110
    step_open = 5
    step_deg = 5

    # 新数据结构：按夹爪编号分组
    grouped_grippers = []  # 每个元素是一个完整夹爪的网格列表
    gripper_params = []  # 每个元素记录夹爪参数 (opening, deg)

    # 生成所有夹爪模型
    for opening in range(min_open, max_open + 1, step_open):
        gripper = GripperModel(
            finger_thickness=finger_thickness,
            finger_width=finger_width,
            finger_length=finger_length,
            opening=opening,
            base_depth=base_depth,
            base_color=base_color,
            finger_color=finger_color,
            axis_size=axis_size
        )
        base_meshes = gripper.get_meshes()  # [finger1, finger2, base, axis]

        for deg in range(0, 180, step_deg):
            angle_rad = np.deg2rad(deg)
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])

            # 为当前姿态创建新的网格组
            current_gripper = []
            for mesh in base_meshes:
                m = copy.deepcopy(mesh)
                m.rotate(R, center=(0, 0, 0))
                current_gripper.append(m)

            # 保存分组和参数
            grouped_grippers.append(current_gripper)
            gripper_params.append((opening, deg))

    # 示例1: 可视化所有夹爪（平面列表）
    # all_meshes_flat = [mesh for group in grouped_grippers for mesh in group]
    # o3d.visualization.draw_geometries(
    #     all_meshes_flat,
    #     window_name='All Grippers',
    #     width=1280,
    #     height=800
    # )

    # 示例2: 可视化特定编号的夹爪（如编号0）
    # gripper_id = 0
    # o3d.visualization.draw_geometries(
    #     grouped_grippers[gripper_id],
    #     window_name=f'Gripper #{gripper_id} (Opening={gripper_params[gripper_id][0]}°, Rotation={gripper_params[gripper_id][1]}°)'
    # )

    return grouped_grippers, gripper_params  # 返回分组数据供外部使用


if __name__ == '__main__':
    # gripper_demo()

    gripper_groups, params = gripper_demo_with_index()

    # 单独操作第10个夹爪模型
    gripper_id = 200
    single_gripper = gripper_groups[gripper_id]  # 包含4个网格的列表
    opening, deg = params[gripper_id]

    # 可视化这个夹爪
    o3d.visualization.draw_geometries(
        single_gripper,
        window_name=f"Gripper #{gripper_id} (Open={opening}mm, Rot={deg}°)"
    )

    # gripper_dict = gripper_demo1(show_all=False, show_index=420)
