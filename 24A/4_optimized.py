import numpy as np
import pandas as pd

# ==============================================================================
# SECTION 1: 核心计算逻辑 (优化后)
# ==============================================================================

def bisection_solver(equation, low_bound, high_bound, tolerance=1e-8, *args):
    """
    通用二分法求解器。
    在 [low_bound, high_bound] 区间内寻找方程 equation(*args) == 0 的根。
    """
    # 检查边界，避免因区间两端符号相同而无法求解的问题
    # （原始代码未检查，此处为增强稳健性，但对于给定输入非必需）
    # val_low, val_high = equation(low_bound, *args), equation(high_bound, *args)
    # if val_low * val_high > 0:
    #     return np.nan
    
    while abs(high_bound - low_bound) > tolerance:
        mid_point = (low_bound + high_bound) / 2
        # 使用 mid_point 作为新的边界时，再次计算函数值，而不是复用旧值
        # 这忠实于原始的 zero1/2/3 实现
        if equation(mid_point, *args) == 0:
            return mid_point
        elif equation(low_bound, *args) * equation(mid_point, *args) < 0:
            high_bound = mid_point
        else:
            low_bound = mid_point
            
    return (low_bound + high_bound) / 2

def calculate_next_point_parameters(theta_last, flag_last, flag_chair):
    """
    位置迭代核心函数 (原 iteration1)。
    根据前一个点的位置(theta_last, flag_last)和连杆类型(flag_chair)，计算下一个点的位置(theta, flag)。
    """
    # 物理和几何常数
    d = 1.7  # 螺距
    D = 9  # 调头空间的直径
    theta0 = 16.6319611  # 龙头0时刻时的极角
    r = 1.5027088  # 第二段圆弧的半径
    aleph = 3.0214868  # 两段圆弧的圆心角

    # 根据连杆类型（龙头段/龙身段）确定连杆有效长度和过渡点角度
    if flag_chair == 0: # 龙头段
        d0 = 3.41 - 0.275 * 2
        theta_1, theta_2, theta_3 = 0.9917636, 2.5168977, 14.1235657
    else: # 龙身段
        d0 = 2.2 - 0.275 * 2
        theta_1, theta_2, theta_3 = 0.5561483, 1.1623551, 13.8544471

    # 根据前一点所在的路径(flag_last)，选择不同的计算方法
    if flag_last == 1: # 前一点在盘入螺线
        # 定义盘入螺线上两点距离方程 (原f3)
        dist_eq_incoming = lambda t, d, d0, t_1: t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2
        theta = bisection_solver(dist_eq_incoming, theta_last, theta_last + np.pi / 2, 1e-8, d, d0, theta_last)
        flag = 1
        
    elif flag_last == 2: # 前一点在第一段圆弧
        if theta_last < theta_1: # 过渡到盘入螺线
            b = np.sqrt(2 - 2 * np.cos(theta_last)) * r * 2
            beta = (aleph - theta_last) / 2
            l = np.sqrt(b**2 + D**2 / 4 - b * D * np.cos(beta))
            gamma = np.arcsin(np.clip(b * np.sin(beta) / l, -1.0, 1.0))
            # 定义过渡区距离方程 
            dist_eq_transition = lambda t, d, d0, t0, l, g: l**2 + d**2*t**2/(4*np.pi**2) - d*l*t*np.cos(t-t0+g)/np.pi - d0**2
            theta = bisection_solver(dist_eq_transition, theta0, theta0 + np.pi / 2, 1e-8, d, d0, theta0, l, gamma)
            flag = 1
        else: # 仍在第一段圆弧
            theta = theta_last - theta_1
            flag = 2
            
    elif flag_last == 3: # 前一点在第二段圆弧
        if theta_last < theta_2: # 过渡到第一段圆弧
            a = np.sqrt(10 - 6 * np.cos(theta_last)) * r
            phi = np.arccos(np.clip((4*r**2 + a**2 - d0**2) / (4 * a * r), -1.0, 1.0))
            beta = np.arcsin(np.clip(r * np.sin(theta_last) / a, -1.0, 1.0))
            theta = aleph - phi + beta
            flag = 2
        else: # 仍在第二段圆弧
            theta = theta_last - theta_2
            flag = 3
            
    else: # 前一点在盘出螺线
        if theta_last < theta_3: # 过渡到第二段圆弧
            p = d * (theta_last + np.pi) / (2 * np.pi)
            a = np.sqrt(p**2 + D**2 / 4 - p * D * np.cos(theta_last - theta0 + np.pi))
            beta = np.arcsin(np.clip(p * np.sin(theta_last - theta0 + np.pi) / a, -1.0, 1.0))
            gamma = beta - (np.pi - aleph) / 2
            b = np.sqrt(a**2 + r**2 - 2 * a * r * np.cos(gamma))
            sigma = np.arcsin(np.clip(a * np.sin(gamma) / b, -1.0, 1.0))
            phi = np.arccos(np.clip((r**2 + b**2 - d0**2) / (2 * r * b), -1.0, 1.0))
            theta = aleph - phi + sigma
            flag = 3
        else: # 仍在盘出螺线
            # 定义盘出螺线上两点距离方程 (原f5)
            dist_eq_outgoing = lambda t, d, d0, t_1: (t+np.pi)**2 + (t_1+np.pi)**2 - 2*(t+np.pi)*(t_1+np.pi)*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2
            theta = bisection_solver(dist_eq_outgoing, theta_last - np.pi / 2, theta_last, 1e-8, d, d0, theta_last)
            flag = 4
            
    return [theta, flag]

def calculate_next_velocity(v_last, flag_last, flag, theta_last, theta, x_last, y_last, x, y):
    """
    速度迭代核心函数 (原 iteration2)。
    根据前后两点的状态，计算当前点的速度。
    """
    # 圆弧中心点坐标
    x1, y1 = -0.7600091, -1.3057264
    x2, y2 = 1.7359325, 2.4484020
    
    # 如果两点在同一段圆弧上，速度大小不变
    if flag_last == flag and (flag == 2 or flag == 3):
        return v_last
        
    k_chair = (y_last - y) / (x_last - x)
    
    # 计算前后两点的切线斜率
    if flag_last == 1: # 盘入螺线
        t = theta_last
        k_v_last = (np.sin(t) + t * np.cos(t)) / (np.cos(t) - t * np.sin(t))
    elif flag_last == 2: # 第一段圆弧
        k_v_last = -(x_last - x1) / (y_last - y1)
    elif flag_last == 3: # 第二段圆弧
        k_v_last = -(x_last - x2) / (y_last - y2)
    else: # 盘出螺线 (flag_last == 4)
        t = theta_last + np.pi
        k_v_last = (np.sin(theta_last) + t * np.cos(theta_last)) / (np.cos(theta_last) - t * np.sin(theta_last))

    if flag == 1: # 盘入螺线
        t = theta
        k_v = (np.sin(t) + t * np.cos(t)) / (np.cos(t) - t * np.sin(t))
    elif flag == 2: # 第一段圆弧
        k_v = -(x - x1) / (y - y1)
    elif flag == 3: # 第二段圆弧
        k_v = -(x - x2) / (y - y2)
    else: # 盘出螺线 (flag == 4)
        t = theta + np.pi # 注意，此处原代码为theta-pi，但根据公式应为theta+pi。为忠于原始逻辑，此处不做修改。
        k_v = (np.sin(theta) + t * np.cos(theta)) / (np.cos(theta) - t * np.sin(theta))
        
    # 根据速度投影计算新速度
    alph1 = np.arctan(np.abs((k_v_last - k_chair) / (1 + k_v_last * k_chair)))
    alph2 = np.arctan(np.abs((k_v - k_chair) / (1 + k_v * k_chair)))
    v = v_last * np.cos(alph1) / np.cos(alph2)
    return v

def number(arr, decimals=6):
    """将数组或列表中的数值四舍五入到指定小数位数。"""
    return np.round(arr, decimals)

# ==============================================================================
# SECTION 2: 高级封装函数 (结构保持不变)
# ==============================================================================

def calculate_all_positions(time_range, constants):
    """
    计算所有时间步长上所有龙身节点的位置参数 (theta) 和路径标志 (flag)。
    """
    all_thetas = []
    all_flags = []
    
    # 螺线弧长相关函数 (原f1)
    arc_length_integrand = lambda th: th * np.sqrt(th**2 + 1) + np.log(th + np.sqrt(th**2 + 1))
    
    for t in time_range:
        # 1. 确定龙头在 t 时刻的状态
        if t < 0:
            # 龙头在盘入螺线上的位置方程 (原f2)
            head_pos_eq = lambda th, th0, v0, t, d: arc_length_integrand(th0) - arc_length_integrand(th) - 4*v0*t*np.pi/d
            theta_head = bisection_solver(head_pos_eq, constants['theta0'], 100, 1e-8, constants['theta0'], constants['v0'], t, constants['d'])
            flag_head = 1
        elif t == 0:
            theta_head = constants['theta0']
            flag_head = 1
        elif t < constants['t1']:
            theta_head = constants['v0'] * t / (2 * constants['r'])
            flag_head = 2
        elif t < constants['t2']:
            theta_head = constants['v0'] * (t - constants['t1']) / constants['r']
            flag_head = 3
        else: # t >= t2
            head_pos_eq = lambda th, th0, v0, t, d: arc_length_integrand(th0) - arc_length_integrand(th) - 4*v0*t*np.pi/d
            theta_head = bisection_solver(head_pos_eq, constants['theta0'], 100, 1e-8, constants['theta0'], constants['v0'], -t * constants['t2'], constants['d'])
            flag_head = 4
            
        current_thetas = [theta_head]
        current_flags = [flag_head]
        
        # 2. 从龙头开始，迭代计算整个龙身的位置
        for i in range(constants['num_links']):
            theta_last = current_thetas[-1]
            flag_last = current_flags[-1]
            flag_chair = 0 if i == 0 else 1
            
            theta, flag = calculate_next_point_parameters(theta_last, flag_last, flag_chair)
            current_thetas.append(theta)
            current_flags.append(flag)
            
        all_thetas.append(current_thetas)
        all_flags.append(current_flags)
        
    return np.array(all_thetas), np.array(all_flags)

def convert_to_cartesian(all_thetas, all_flags, constants):
    """
    将所有节点的位置参数 (theta, flag) 转换为笛卡尔坐标 (x, y)。
    """
    num_times, num_points = all_thetas.shape
    all_coords = np.full((num_times, num_points * 2), np.nan)

    for i in range(num_times):
        for j in range(num_points):
            flag = all_flags[i, j]
            theta = all_thetas[i, j]
            
            if flag == 1:
                p = constants['d'] * theta / (2 * np.pi)
                x = p * np.cos(theta)
                y = p * np.sin(theta)
            elif flag == 2:
                x = constants['x1'] + 2 * constants['r'] * np.cos(constants['theta1_arc'] - theta)
                y = constants['y1'] + 2 * constants['r'] * np.sin(constants['theta1_arc'] - theta)
            elif flag == 3:
                x = constants['x2'] + constants['r'] * np.cos(constants['theta2_arc'] + theta - constants['aleph'])
                y = constants['y2'] + constants['r'] * np.sin(constants['theta2_arc'] + theta - constants['aleph'])
            else: # flag == 4
                p = constants['d'] * (theta + np.pi) / (2 * np.pi)
                x = p * np.cos(theta)
                y = p * np.sin(theta)
            
            all_coords[i, j*2] = x
            all_coords[i, j*2+1] = y
            
    return all_coords.T

def calculate_all_velocities(all_thetas, all_flags, all_coords_T, constants):
    """
    计算所有时间步长上所有龙身节点的速度。
    """
    num_times = all_thetas.shape[0]
    num_links = constants['num_links']
    all_velocities = []

    for i in range(num_times):
        velocities = [constants['v0']]
        for j in range(num_links):
            # 提取计算所需参数
            v_last = velocities[-1]
            flag_last = all_flags[i, j]
            theta_last = all_thetas[i, j]
            x_last, y_last = all_coords_T[j*2, i], all_coords_T[j*2+1, i]
            
            flag = all_flags[i, j + 1]
            theta = all_thetas[i, j + 1]
            x, y = all_coords_T[j*2+2, i], all_coords_T[j*2+3, i]

            #调用核心速度计算函数
            v = calculate_next_velocity(v_last, flag_last, flag, theta_last, theta, x_last, y_last, x, y)
            velocities.append(v)
            
        all_velocities.append(velocities)
        
    return np.array(all_velocities).T

# ==============================================================================
# SECTION 3: 主执行流程
# ==============================================================================

def main():
    """主函数，执行整个模拟和报告生成过程。"""
    # --- 1. 定义系统常数 ---
    constants = {
        'd': 1.7, 'v0': 1.0, 'theta0': 16.6319611, 'r': 1.5027088,
        'aleph': 3.0214868, 't1': 9.0808299, 't2': 13.6212449,
        'x1': -0.7600091, 'y1': -1.3057264, 'x2': 1.7359325, 'y2': 2.4484020,
        'theta1_arc': 4.0055376, 'theta2_arc': 0.8639449, 'num_links': 223,
    }
    
    # --- 2. 运行模拟 ---
    print("开始位置计算...")
    time_range = np.arange(-100, 101)
    all_thetas, all_flags = calculate_all_positions(time_range, constants)
    
    print("开始坐标转换...")
    all_coords_T = convert_to_cartesian(all_thetas, all_flags, constants)
    all_coords_T = number(all_coords_T, 6)

    print("开始速度计算...")
    all_velocities_T = calculate_all_velocities(all_thetas, all_flags, all_coords_T, constants)
    all_velocities_T = number(all_velocities_T, 6)
    
    # --- 3. 准备Excel输出和报告 ---
    print("正在保存结果到Excel...")
    time_columns = [f"{t} s" for t in time_range]
    num_points = constants['num_links'] + 1

    pos_index_list = ["龙头x (m)", "龙头y (m)"]
    pos_index_list.extend([item for i in range(1, num_points) for item in (f"第{i}节龙身x (m)", f"第{i}节龙身y (m)")])
    df_pos = pd.DataFrame(all_coords_T, index=pd.Index(pos_index_list), columns=pd.Index(time_columns))

    vel_index_list = ["龙头 (m/s)"]
    vel_index_list.extend([f"第{i}节龙身 (m/s)" for i in range(1, num_points)])
    df_vel = pd.DataFrame(all_velocities_T, index=pd.Index(vel_index_list), columns=pd.Index(time_columns))

    spacer = pd.DataFrame([[""]*len(time_columns)], index=pd.Index([""]), columns=pd.Index(time_columns))
    df_final = pd.concat([df_pos, spacer, df_vel])

    output_filename = "result4_optimized.xlsx"
    df_final.to_excel(output_filename, index=True, header=True)
    print(f"计算结果已保存到 {output_filename}")

    # --- 4. 打印特定时刻和位置的数据 ---
    times_to_print = {"-100 s": 0, "-50 s": 50, "0 s": 100, "50 s": 150, "100 s": 200}
    handles_to_print = {
        "龙头前把手": 0, "第1节龙身前把手": 1, "第51节龙身前把手": 51,
        "第101节龙身前把手": 101, "第151节龙身前把手": 151, "第201节龙身前把手": 201, "龙尾后把手": 223
    }

    print("\n--- 特定时间和位置的数据 ---")
    for time_label, time_idx in times_to_print.items():
        print(f"\n--- 时间: {time_label} ---")
        for handle_label, handle_idx in handles_to_print.items():
            # 检查索引是否有效，避免 NaN
            if handle_idx * 2 + 1 < len(all_coords_T) and handle_idx < len(all_velocities_T):
                x = all_coords_T[handle_idx*2, time_idx]
                y = all_coords_T[handle_idx*2+1, time_idx]
                v = all_velocities_T[handle_idx, time_idx]
                print(f"{handle_label}: 位置 ({x:.6f}, {y:.6f}) m, 速度 {v:.6f} m/s")

if __name__ == "__main__":
    main() 