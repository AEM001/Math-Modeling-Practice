# 问题4：舞龙队调头运动学分析（修正版伪代码）
# 基于正确代码思路重新设计

import numpy as np
import pandas as pd
from scipy.optimize import fsolve

# ===== 1. 核心数学函数 =====
def spiral_length_integral(theta):
    """
    计算螺线长度积分 - 这是关键的数学工具
    ∫√(ρ² + (dρ/dθ)²) dθ 的解析解
    """
    if theta <= 1e-9:
        return 0
    return theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))

def find_zero_point(func, a, b, tolerance=1e-8, *args):
    """
    通用二分法求零点 - 修正参数传递方式
    完全按照正确代码的实现方式
    """
    # 确保搜索区间有效
    if abs(func(a, *args)) < tolerance:
        return a
    if abs(func(b, *args)) < tolerance:
        return b
        
    # 检查函数值符号是否相反
    fa, fb = func(a, *args), func(b, *args)
    if fa * fb > 0:
        # 如果同号，尝试扩大搜索范围
        return np.nan
    
    while b - a >= tolerance:
        c = (a + b) / 2
        fc = func(c, *args)
        if abs(fc) < tolerance:
            return c
        if func(a, *args) * fc < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

def round_array(arr, decimals=6):
    """数组保留指定小数位数 - 从正确代码复制"""
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr
    """计算螺线上某点的速度斜率（切线斜率）"""
    numerator = np.sin(theta) + theta * np.cos(theta)
    denominator = np.cos(theta) - theta * np.sin(theta)
    return numerator / denominator if abs(denominator) > 1e-9 else np.inf

# ===== 2. 系统参数定义 =====
class DragonSystemParams:
    def __init__(self):
        # 基本物理参数
        self.d = 1.7            # 螺距 (m)
        self.v0 = 1.0           # 龙头速度 (m/s)
        self.num_segments = 223 # 龙身节数
        self.num_handles = 224  # 把手总数（龙头+223节龙身）
        
        # 关键几何参数（预计算得到）
        self.theta0 = 16.6319611    # 龙头在t=0时的极角
        self.r = 1.5027088          # 第二段圆弧半径
        self.aleph = 3.0214868      # 两段圆弧的圆心角
        self.t1 = 9.0808299         # 龙头到达第二段圆弧的时刻
        self.t2 = 13.6212449        # 龙头到达盘出螺线的时刻
        
        # 圆心坐标（精确值从正确代码获取）
        self.x1, self.y1 = -0.7600091, -1.3057264  # 第一段圆弧圆心
        self.x2, self.y2 = 1.7359325, 2.4484020    # 第二段圆弧圆心
        self.theta1 = 4.0055376     # 第一段圆弧进入角度
        self.theta2 = 0.8639449     # 第二段圆弧离开角度
        
        # 龙身长度
        self.segment_lengths = np.full(self.num_segments, 2.2 - 0.275 * 2)
        self.segment_lengths[0] = 3.41 - 0.275 * 2  # 龙头特殊长度

# ===== 3. 龙头位置计算 =====
def calculate_head_position(t, params):
    """
    根据时间t计算龙头前把手的位置参数
    返回: (theta, flag)
    flag: 1=盘入螺线, 2=第一段圆弧, 3=第二段圆弧, 4=盘出螺线
    """
    if t < 0:
        # 阶段1：盘入螺线 - 关键修正：直接使用t而不是-t
        def head_equation(theta):
            return (spiral_length_integral(params.theta0) - 
                   spiral_length_integral(theta) - 
                   4 * params.v0 * t * np.pi / params.d)  # 直接使用t
        
        # 修正搜索范围：t<0时theta应该大于theta0
        theta_head = find_zero_point(head_equation, params.theta0, params.theta0 + 50)
        return theta_head, 1
        
    elif t == 0:
        # 边界条件：刚进入第一段圆弧，但仍在盘入螺线上
        return params.theta0, 1
        
    elif t < params.t1:
        # 阶段2：第一段圆弧 - 简单的角速度关系
        theta_head = params.v0 * t / (2 * params.r)
        return theta_head, 2
        
    elif t < params.t2:
        # 阶段3：第二段圆弧
        theta_head = params.v0 * (t - params.t1) / params.r
        return theta_head, 3
        
    else:
        # 阶段4：盘出螺线 - 关键修正：使用-t+t2而不是t-t2
        def head_equation_out(theta):
            return (spiral_length_integral(params.theta0) - 
                   spiral_length_integral(theta) - 
                   4 * params.v0 * (-t + params.t2) * np.pi / params.d)
        
        # 在盘出螺线上搜索，然后减去π
        theta_head = find_zero_point(head_equation_out, params.theta0, 100) - np.pi
        return theta_head, 4

# ===== 4. 位置递推核心算法 =====
def position_iteration(theta_prev, flag_prev, segment_index, params):
    """
    核心递推函数：根据前一个把手位置计算下一个把手位置
    完全按照正确代码的逻辑重新实现
    """
    # 确定当前段的长度
    if segment_index == 0:
        d0 = 3.41 - 0.275 * 2  # 龙头板长
        theta_1, theta_2, theta_3 = 0.9917636, 2.5168977, 14.1235657
    else:
        d0 = 2.2 - 0.275 * 2   # 龙身板长
        theta_1, theta_2, theta_3 = 0.5561483, 1.1623551, 13.8544471
    
    # 根据前把手所在阶段分情况处理
    if flag_prev == 1:  # 前把手在盘入螺线
        # 几何约束：两点间距离等于龙身长度
        def spiral_constraint(theta):
            return (theta**2 + theta_prev**2 - 
                   2*theta*theta_prev*np.cos(theta - theta_prev) - 
                   4*np.pi**2*d0**2/params.d**2)
        
        theta = find_zero_point(spiral_constraint, theta_prev, theta_prev + np.pi/2)
        return theta, 1
        
    elif flag_prev == 2:  # 前把手在第一段圆弧
        if theta_prev < theta_1:
            # 跨阶段：第一段圆弧到盘入螺线
            b = np.sqrt(2-2*np.cos(theta_prev)) * params.r * 2
            beta = (params.aleph - theta_prev) / 2
            l = np.sqrt(b**2 + 9.0**2/4 - b*9.0*np.cos(beta))
            gamma = np.arcsin(b*np.sin(beta)/l)
            
            def transition_constraint(theta):
                return (l**2 + params.d**2*theta**2/(4*np.pi**2) - 
                       params.d*l*theta*np.cos(theta-params.theta0+gamma)/np.pi - d0**2)
            
            theta = find_zero_point(transition_constraint, params.theta0, params.theta0 + np.pi/2)
            return theta, 1
        else:
            # 同阶段：第一段圆弧内部
            theta = theta_prev - theta_1
            return theta, 2
            
    elif flag_prev == 3:  # 前把手在第二段圆弧
        if theta_prev < theta_2:
            # 跨阶段：第二段圆弧到第一段圆弧
            a = np.sqrt(10-6*np.cos(theta_prev)) * params.r
            phi = np.arccos((4*params.r**2+a**2-d0**2)/(4*a*params.r))
            beta = np.arcsin(params.r*np.sin(theta_prev)/a)
            theta = params.aleph - phi + beta
            return theta, 2
        else:
            # 同阶段：第二段圆弧内部
            theta = theta_prev - theta_2
            return theta, 3
            
    else:  # flag_prev == 4，前把手在盘出螺线
        if theta_prev < theta_3:
            # 跨阶段：盘出螺线到第二段圆弧
            p = params.d * (theta_prev + np.pi) / (2 * np.pi)
            a = np.sqrt(p**2 + 9.0**2/4 - p*9.0*np.cos(theta_prev-params.theta0+np.pi))
            beta = np.arcsin(p*np.sin(theta_prev-params.theta0+np.pi)/a)
            gamma = beta - (np.pi-params.aleph)/2
            b = np.sqrt(a**2 + params.r**2 - 2*a*params.r*np.cos(gamma))
            sigma = np.arcsin(a*np.sin(gamma)/b)
            phi = np.arccos((params.r**2+b**2-d0**2)/(2*params.r*b))
            theta = params.aleph - phi + sigma
            return theta, 3
        else:
            # 同阶段：盘出螺线内部
            def spiral_out_constraint(theta):
                t = theta + np.pi
                t_1 = theta_prev + np.pi
                return (t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 
                       4*np.pi**2*d0**2/params.d**2)
            
            theta = find_zero_point(spiral_out_constraint, theta_prev - np.pi/2, theta_prev)
            return theta, 4

# ===== 5. 坐标转换 =====
def theta_to_coordinates(theta, flag, params):
    """根据theta和flag转换为笛卡尔坐标"""
    if flag == 1:  # 盘入螺线
        rho = params.d * theta / (2 * np.pi)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
    elif flag == 2:  # 第一段圆弧
        x = params.x1 + 2 * params.r * np.cos(params.theta1 - theta)
        y = params.y1 + 2 * params.r * np.sin(params.theta1 - theta)
    elif flag == 3:  # 第二段圆弧
        x = params.x2 + params.r * np.cos(params.theta2 + theta - params.aleph)
        y = params.y2 + params.r * np.sin(params.theta2 + theta - params.aleph)
    else:  # flag == 4，盘出螺线
        rho = params.d * (theta + np.pi) / (2 * np.pi)
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
    
    return x, y

# ===== 6. 速度计算 =====
def velocity_iteration(v_prev, flag_prev, flag_curr, theta_prev, theta_curr, 
                      pos_prev, pos_curr, params):
    """
    速度递推计算 - 完全按照正确代码的逻辑重新实现
    """
    x_prev, y_prev = pos_prev
    x_curr, y_curr = pos_curr
    
    # 精确的圆心坐标（从正确代码中获取）
    x1, y1 = -0.7600091, -1.3057264  # 第一段圆弧圆心
    x2, y2 = 1.7359325, 2.4484020    # 第二段圆弧圆心
    
    # 计算龙身的斜率
    if abs(x_prev - x_curr) < 1e-9:
        k_board = np.inf
    else:
        k_board = (y_prev - y_curr) / (x_prev - x_curr)
    
    # 根据不同阶段组合计算速度斜率
    if flag_prev == 1 and flag_curr == 1:  # 都在盘入螺线
        k_v_prev = velocity_slope_on_spiral(theta_prev)
        k_v_curr = velocity_slope_on_spiral(theta_curr)
    elif flag_prev == 2 and flag_curr == 1:  # 第一段圆弧到盘入螺线
        k_v_prev = -(x_prev - x1) / (y_prev - y1)
        k_v_curr = velocity_slope_on_spiral(theta_curr)
    elif flag_prev == 2 and flag_curr == 2:  # 都在第一段圆弧
        return v_prev  # 同圆弧内速度相等
    elif flag_prev == 3 and flag_curr == 2:  # 第二段圆弧到第一段圆弧
        k_v_prev = -(x_prev - x2) / (y_prev - y2)
        k_v_curr = -(x_curr - x1) / (y_curr - y1)
    elif flag_prev == 3 and flag_curr == 3:  # 都在第二段圆弧
        return v_prev  # 同圆弧内速度相等
    elif flag_prev == 4 and flag_curr == 3:  # 盘出螺线到第二段圆弧
        k_v_prev = velocity_slope_on_spiral(theta_prev + np.pi)
        k_v_curr = -(x_curr - x2) / (y_curr - y2)
    else:  # 都在盘出螺线 (flag_prev == 4 and flag_curr == 4)
        k_v_prev = velocity_slope_on_spiral(theta_prev + np.pi)
        k_v_curr = velocity_slope_on_spiral(theta_curr + np.pi)
    
    # 计算角度和速度传递
    try:
        if np.isinf(k_board):
            # 处理垂直龙身的特殊情况
            if np.isinf(k_v_prev):
                angle1 = 0
            else:
                angle1 = np.pi/2 - np.arctan(k_v_prev)
            
            if np.isinf(k_v_curr):
                angle2 = 0
            else:
                angle2 = np.pi/2 - np.arctan(k_v_curr)
        else:
            # 一般情况：使用角度差公式
            angle1 = np.arctan(abs((k_v_prev - k_board) / (1 + k_v_prev * k_board)))
            angle2 = np.arctan(abs((k_v_curr - k_board) / (1 + k_v_curr * k_board)))
        
        # 速度传递公式
        return v_prev * np.cos(angle1) / np.cos(angle2)
    except:
        # 异常情况下保持原速度
        return v_prev

# ===== 7. 主模拟函数 =====
def simulate_dragon_dance():
    """主模拟函数 - 按照正确代码的结构重新实现"""
    params = DragonSystemParams()
    
    # 时间范围：-100s到100s
    time_points = np.arange(-100, 101)
    
    # 存储所有时刻的theta和flag
    all_positions_theta = []
    all_positions_flag = []
    
    print("开始计算所有时刻的位置参数...")
    
    # 第一步：计算所有时刻的theta和flag
    for t in time_points:
        # 计算龙头位置
        head_theta, head_flag = calculate_head_position(t, params)
        
        # 初始化当前时刻的数组
        current_thetas = [head_theta]
        current_flags = [head_flag]
        
        # 递推计算所有把手位置
        for i in range(params.num_segments):
            try:
                theta_next, flag_next = position_iteration(
                    current_thetas[-1], current_flags[-1], i, params)
                current_thetas.append(theta_next)
                current_flags.append(flag_next)
            except:
                print(f"计算第{i}个把手时出错，在t={t}s")
                # 使用默认值
                current_thetas.append(current_thetas[-1])
                current_flags.append(current_flags[-1])
        
        all_positions_theta.append(current_thetas)
        all_positions_flag.append(current_flags)
    
    # 转换为numpy数组
    all_positions_theta = np.array(all_positions_theta)
    all_positions_flag = np.array(all_positions_flag)
    
    print("开始计算坐标...")
    
    # 第二步：计算坐标
    all_coordinates = []
    for i, t in enumerate(time_points):
        coords = []
        for j in range(params.num_handles):
            theta = all_positions_theta[i, j]
            flag = all_positions_flag[i, j]
            x, y = theta_to_coordinates(theta, flag, params)
            coords.extend([x, y])
        all_coordinates.append(coords)
    
    # 转换并四舍五入
    all_coordinates = np.array(all_coordinates).T
    all_coordinates = round_array(all_coordinates, 6)
    
    print("开始计算速度...")
    
    # 第三步：计算速度
    all_velocities = []
    for i, t in enumerate(time_points):
        velocities = [params.v0]  # 龙头速度恒定
        
        for j in range(params.num_segments):
            # 获取相关参数
            flag_prev = all_positions_flag[i, j]
            theta_prev = all_positions_theta[i, j]
            flag_curr = all_positions_flag[i, j + 1]
            theta_curr = all_positions_theta[i, j + 1]
            
            # 获取坐标
            x_prev = all_coordinates[j * 2, i]
            y_prev = all_coordinates[j * 2 + 1, i]
            x_curr = all_coordinates[j * 2 + 2, i]
            y_curr = all_coordinates[j * 2 + 3, i]
            
            # 计算速度
            v = velocity_iteration(velocities[-1], flag_prev, flag_curr, 
                                 theta_prev, theta_curr, 
                                 (x_prev, y_prev), (x_curr, y_curr), params)
            velocities.append(v)
        
        all_velocities.append(velocities)
    
    # 转换并四舍五入
    all_velocities = np.array(all_velocities).T
    all_velocities = round_array(all_velocities, 6)
    
    # 保存结果
    save_results(all_coordinates, all_velocities, time_points)
    
    # 输出关键时刻的详细数据
    output_key_results(all_coordinates, all_velocities, time_points)
    
    print("模拟完成！")

def save_results(coordinates, velocities, time_points):
    """保存结果到Excel文件"""
    # 转换数据格式
    coords_array = np.array(coordinates).T
    velocities_array = np.array(velocities).T
    
    # 保存坐标
    df_coords = pd.DataFrame(coords_array)
    df_coords.to_excel("result4_coordinates.xlsx", index=False)
    
    # 保存速度
    df_velocities = pd.DataFrame(velocities_array)
    df_velocities.to_excel("result4_velocities.xlsx", index=False)

def output_key_results(coordinates, velocities, time_points):
    """输出关键时刻和位置的详细数据"""
    print("\n" + "="*60)
    print("关键时刻位置和速度报告")
    print("="*60)
    
    # 关键时刻索引
    key_times = {"-100s": 0, "-50s": 50, "0s": 100, "50s": 150, "100s": 200}
    
    # 关键把手索引
    key_handles = {
        "龙头前把手": 0, "第1节龙身前把手": 1, "第51节龙身前把手": 51,
        "第101节龙身前把手": 101, "第151节龙身前把手": 151, 
        "第201节龙身前把手": 201, "龙尾后把手": 223
    }
    
    for time_label, time_idx in key_times.items():
        print(f"\n--- 时间: {time_label} ---")
        for handle_label, handle_idx in key_handles.items():
            x = coordinates[time_idx][handle_idx * 2]
            y = coordinates[time_idx][handle_idx * 2 + 1]
            v = velocities[time_idx][handle_idx]
            print(f"{handle_label}: 位置({x:.6f}, {y:.6f})m, 速度{v:.6f}m/s")

# ===== 8. 程序执行 =====
if __name__ == "__main__":
    simulate_dragon_dance()