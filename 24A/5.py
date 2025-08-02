import numpy as np
from tqdm import tqdm

# ==============================================================================
# SECTION 0: 统一的系统常量定义（与前四问保持一致）
# ==============================================================================
NUM_LINKS = 223  # 连杆数量
NUM_POINTS = NUM_LINKS + 1  # 铰接点数量（224个）
HINGE_OFFSET = 0.275  # 铰接点距连杆端部的偏移量 (m)

# 连杆物理长度
physical_link_lengths = np.full(NUM_LINKS, 2.20)
physical_link_lengths[0] = 3.41  # 第一个连杆（龙头段）特殊长度

# 铰接点之间的有效长度
effective_link_lengths = physical_link_lengths - (2 * HINGE_OFFSET)

# ==============================================================================
# SECTION 1: 核心数学函数（从问题4复用）
# ==============================================================================

def f1(theta):
    """螺线弧长积分函数"""
    return theta * np.sqrt(theta**2 + 1) + np.log(theta + np.sqrt(theta**2 + 1))

def f2(theta, theta0, v0, t, d):
    """龙头位置方程"""
    return f1(theta0) - f1(theta) - 4*v0*t*np.pi/d

def f3(theta, d, d0, theta_last):
    """盘入螺线上两点距离方程"""
    t = theta
    t_1 = theta_last
    return t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2

def f4(theta):
    """螺线切线斜率"""
    t = theta
    return (np.sin(t) + t*np.cos(t)) / (np.cos(t) - t*np.sin(t))

def f5(theta, d, d0, theta_last):
    """盘出螺线上两点距离方程"""
    t = theta + np.pi
    t_1 = theta_last + np.pi
    return t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2

def f6(theta, d, d0, theta0, l, gamma):
    """过渡区距离方程"""
    t = theta
    t0 = theta0
    return l**2 + d**2*t**2/(4*np.pi**2) - d*l*t*np.cos(t-t0+gamma)/np.pi - d0**2

# ==============================================================================
# SECTION 2: 二分法求解器
# ==============================================================================

def bisection_solver(f, a, b, eps, *args):
    """通用二分法求解器"""
    while abs(b - a) > eps:
        c = (a + b) / 2
        if f(c, *args) == 0:
            return c
        elif f(a, *args) * f(c, *args) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2

# ==============================================================================
# SECTION 3: 位置和速度计算函数（从问题4复用）
# ==============================================================================

def calculate_next_position(theta_last, flag_last, link_index):
    """位置迭代核心函数"""
    # 系统常数
    d = 1.7
    D = 9
    theta0 = 16.6319611
    r = 1.5027088
    aleph = 3.0214868
    
    # 根据连杆类型确定参数
    if link_index == 0:  # 龙头段
        d0 = effective_link_lengths[0]
        theta_1 = 0.9917636
        theta_2 = 2.5168977
        theta_3 = 14.1235657
    else:  # 龙身段
        d0 = effective_link_lengths[1]
        theta_1 = 0.5561483
        theta_2 = 1.1623551
        theta_3 = 13.8544471
    
    if flag_last == 1:  # 盘入螺线
        theta = bisection_solver(f3, theta_last, theta_last + np.pi/2, 1e-8, d, d0, theta_last)
        flag = 1
    elif flag_last == 2:  # 第一段圆弧
        if theta_last < theta_1:
            b = np.sqrt(2 - 2*np.cos(theta_last)) * r * 2
            beta = (aleph - theta_last) / 2
            l = np.sqrt(b**2 + D**2/4 - b*D*np.cos(beta))
            gamma = np.arcsin(np.clip(b*np.sin(beta)/l, -1.0, 1.0))
            theta = bisection_solver(f6, theta0, theta0 + np.pi/2, 1e-8, d, d0, theta0, l, gamma)
            flag = 1
        else:
            theta = theta_last - theta_1
            flag = 2
    elif flag_last == 3:  # 第二段圆弧
        if theta_last < theta_2:
            a = np.sqrt(10 - 6*np.cos(theta_last)) * r
            phi = np.arccos(np.clip((4*r**2 + a**2 - d0**2)/(4*a*r), -1.0, 1.0))
            beta = np.arcsin(np.clip(r*np.sin(theta_last)/a, -1.0, 1.0))
            theta = aleph - phi + beta
            flag = 2
        else:
            theta = theta_last - theta_2
            flag = 3
    else:  # flag_last == 4, 盘出螺线
        if theta_last < theta_3:
            p = d * (theta_last + np.pi) / (2*np.pi)
            a = np.sqrt(p**2 + D**2/4 - p*D*np.cos(theta_last - theta0 + np.pi))
            sin_arg = p*np.sin(theta_last - theta0 + np.pi)/a
            sin_arg = np.clip(sin_arg, -1.0, 1.0)
            beta = np.arcsin(sin_arg)
            gamma = beta - (np.pi - aleph)/2
            b = np.sqrt(a**2 + r**2 - 2*a*r*np.cos(gamma))
            sigma = np.arcsin(np.clip(a*np.sin(gamma)/b, -1.0, 1.0))
            phi = np.arccos(np.clip((r**2 + b**2 - d0**2)/(2*r*b), -1.0, 1.0))
            theta = aleph - phi + sigma
            flag = 3
        else:
            theta = bisection_solver(f5, theta_last - np.pi/2, theta_last, 1e-8, d, d0, theta_last)
            flag = 4
    
    return theta, flag

def get_coordinates(theta, flag):
    """根据theta和flag计算坐标"""
    d = 1.7
    r = 1.5027088
    aleph = 3.0214868
    x1 = -0.7600091
    y1 = -1.3057264
    x2 = 1.7359325
    y2 = 2.4484020
    theta1_arc = 4.0055376
    theta2_arc = 0.8639449
    
    if flag == 1:  # 盘入螺线
        p = d * theta / (2 * np.pi)
        x = p * np.cos(theta)
        y = p * np.sin(theta)
    elif flag == 2:  # 第一段圆弧
        x = x1 + 2 * r * np.cos(theta1_arc - theta)
        y = y1 + 2 * r * np.sin(theta1_arc - theta)
    elif flag == 3:  # 第二段圆弧
        x = x2 + r * np.cos(theta2_arc + theta - aleph)
        y = y2 + r * np.sin(theta2_arc + theta - aleph)
    else:  # flag == 4, 盘出螺线
        p = d * (theta + np.pi) / (2 * np.pi)
        x = p * np.cos(theta)
        y = p * np.sin(theta)
    
    return x, y

def calculate_velocity(v_last, flag_last, flag, theta_last, theta, x_last, y_last, x, y):
    """速度迭代核心函数"""
    x1 = -0.7600091
    y1 = -1.3057264
    x2 = 1.7359325
    y2 = 2.4484020
    
    # 避免除零
    if abs(x_last - x) < 1e-10:
        return v_last
    
    k_chair = (y_last - y) / (x_last - x)
    v = -1
    
    if flag_last == 1 and flag == 1:
        k_v_last = f4(theta_last)
        k_v = f4(theta)
    elif flag_last == 2 and flag == 1:
        k_v_last = -(x_last - x1) / (y_last - y1)
        k_v = f4(theta)
    elif flag_last == 2 and flag == 2:
        v = v_last
    elif flag_last == 3 and flag == 2:
        k_v_last = -(x_last - x2) / (y_last - y2)
        k_v = -(x - x1) / (y - y1)
    elif flag_last == 3 and flag == 3:
        v = v_last
    elif flag_last == 4 and flag == 3:
        theta_last_adjusted = theta_last + np.pi
        k_v_last = f4(theta_last_adjusted)
        k_v = -(x - x2) / (y - y2)
    else:  # flag_last == 4 and flag == 4
        theta_last_adjusted = theta_last + np.pi
        theta_adjusted = theta - np.pi
        k_v_last = f4(theta_last_adjusted)
        k_v = f4(theta_adjusted)
    
    if v == -1:
        denom1 = 1 + k_v_last * k_chair
        denom2 = 1 + k_v * k_chair
        if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
            v = v_last
        else:
            alph1 = np.arctan(np.abs((k_v_last - k_chair) / denom1))
            alph2 = np.arctan(np.abs((k_v - k_chair) / denom2))
            if abs(np.cos(alph2)) < 1e-10:
                v = np.inf
            else:
                v = v_last * np.cos(alph1) / np.cos(alph2)
    
    return v

# ==============================================================================
# SECTION 4: 问题5的优化求解 - 使用三分搜索法
# ==============================================================================

def calculate_max_velocity_at_time(t, v_head=1.0):
    """计算给定时刻所有把手速度的最大值"""
    # 系统常数
    d = 1.7
    theta0 = 16.6319611
    r = 1.5027088
    t1 = 9.0808299
    t2 = 13.6212449
    
    # 确定龙头状态
    if t < 0:
        theta_head = bisection_solver(f2, 1e-6, 100, 1e-8, theta0, v_head, t, d)
        flag_head = 1
    elif t == 0:
        theta_head = theta0
        flag_head = 1
    elif t < t1:
        theta_head = v_head * t / (2 * r)
        flag_head = 2
    elif t < t2:
        theta_head = v_head * (t - t1) / r
        flag_head = 3
    else:
        theta_head = bisection_solver(f2, 1e-6, 100, 1e-8, theta0, v_head, -t + t2, d)
        flag_head = 4
    
    # 计算所有把手的速度
    velocities = [v_head]
    theta_prev = theta_head
    flag_prev = flag_head
    
    for i in range(NUM_LINKS):
        theta_curr, flag_curr = calculate_next_position(theta_prev, flag_prev, i)
        x_prev, y_prev = get_coordinates(theta_prev, flag_prev)
        x_curr, y_curr = get_coordinates(theta_curr, flag_curr)
        
        v_curr = calculate_velocity(velocities[-1], flag_prev, flag_curr, 
                                  theta_prev, theta_curr, x_prev, y_prev, x_curr, y_curr)
        velocities.append(abs(v_curr))  # 取绝对值
        
        theta_prev = theta_curr
        flag_prev = flag_curr
    
    return max(velocities)

def ternary_search(left, right, epsilon=1e-8):
    """三分搜索法寻找最大值"""
    print(f"\n开始三分搜索，初始区间: [{left:.3f}, {right:.3f}]")
    
    iteration = 0
    while right - left > epsilon:
        iteration += 1
        
        # 计算两个三分点
        t1 = left + (right - left) / 3
        t2 = right - (right - left) / 3
        
        # 计算函数值
        f1 = calculate_max_velocity_at_time(t1)
        f2 = calculate_max_velocity_at_time(t2)
        
        if iteration % 10 == 0:  # 每10次迭代显示一次进度
            print(f"迭代 {iteration}: 区间[{left:.6f}, {right:.6f}], "
                  f"f({t1:.6f})={f1:.6f}, f({t2:.6f})={f2:.6f}")
        
        # 更新搜索区间
        if f1 < f2:
            left = t1
        else:
            right = t2
    
    # 返回最终结果
    t_final = (left + right) / 2
    max_v = calculate_max_velocity_at_time(t_final)
    
    print(f"\n三分搜索完成!")
    print(f"迭代次数: {iteration}")
    print(f"最优时间点: t = {t_final:.8f} s")
    print(f"最大速度: {max_v:.8f} m/s")
    
    return t_final, max_v

def optimize_head_velocity():
    """优化求解龙头最大速度"""
    print("="*60)
    print("问题5：龙头最大速度优化")
    print("="*60)
    
    # 第一步：在指定时间范围内粗略搜索
    print("\n第一步：粗略搜索关键时间段...")
    
    # 根据文档，只需要搜索这两个时间段
    time_ranges = [(13, 16), (376, 384)]
    max_velocities = []
    
    for t_start, t_end in time_ranges:
        print(f"\n搜索时间段 [{t_start}, {t_end}] s...")
        time_points = np.arange(t_start, t_end + 0.1, 0.1)
        
        for t in tqdm(time_points, desc=f"扫描 [{t_start}, {t_end}]"):
            try:
                max_v = calculate_max_velocity_at_time(t)
                max_velocities.append((t, max_v))
            except:
                continue
    
    # 找到最大值
    if not max_velocities:
        print("错误：未找到有效的速度值")
        return None
    
    max_velocities.sort(key=lambda x: x[1], reverse=True)
    t_rough, v_rough = max_velocities[0]
    
    print(f"\n粗略搜索结果：")
    print(f"最大速度出现在 t = {t_rough:.1f} s")
    print(f"最大速度约为 {v_rough:.6f} m/s")
    
    # 第二步：使用三分搜索精确定位
    print("\n第二步：三分搜索精确定位...")
    
    # 确定精确搜索区间（根据文档，应该在[14, 15]区间）
    if 14 <= t_rough <= 15:
        search_left = 14.0
        search_right = 15.0
    else:
        # 如果不在预期区间，在粗略结果附近搜索
        search_left = t_rough - 0.5
        search_right = t_rough + 0.5
    
    # 执行三分搜索
    t_optimal, max_v_optimal = ternary_search(search_left, search_right)
    
    # 第三步：计算龙头最大速度
    v_limit = 2.0  # 速度限制
    v_head_max = v_limit / max_v_optimal
    
    print("\n" + "="*60)
    print("最终结果：")
    print(f"最优时间点: t = {t_optimal:.8f} s")
    print(f"此时最大速度: {max_v_optimal:.8f} m/s")
    print(f"龙头最大速度: v_max = {v_head_max:.6f} m/s")
    print("="*60)
    
    # 验证结果
    print("\n验证计算...")
    max_v_check = calculate_max_velocity_at_time(t_optimal, v_head_max)
    print(f"验证：当 v_head = {v_head_max:.6f} m/s 时")
    print(f"      最大速度 = {max_v_check:.6f} m/s （应该接近 2.0 m/s）")
    
    return v_head_max, t_optimal, max_v_optimal

if __name__ == "__main__":
    result = optimize_head_velocity()
    if result:
        v_head_max, t_optimal, max_v_at_1 = result
        print(f"\n最终答案：龙头最大行进速度 = {v_head_max:.6f} m/s")