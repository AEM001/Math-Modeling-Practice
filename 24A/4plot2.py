import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif']=['KaiTi']
plt.rcParams['axes.unicode_minus'] = False

# ============ 核心计算函数（从4_1.py整合） ============ 

def spiral_length_integral(theta):
    """计算螺线长度积分"""
    # 避免在theta接近0时出现log(0)警告
    safe_theta = np.maximum(theta, 1e-9)
    return safe_theta * np.sqrt(safe_theta**2 + 1) + np.log(safe_theta + np.sqrt(safe_theta**2 + 1))

def find_zero_point(func, a, b, tolerance=1e-8, *args):
    """通用二分法求零点"""
    # 确保函数在区间端点异号
    fa = func(a, *args)
    fb = func(b, *args)
    if fa * fb >= 0:
        # 如果同号，可能是一个端点就是解，或者范围内无解
        if abs(fa) < tolerance: return a
        if abs(fb) < tolerance: return b
        # print(f"警告: find_zero_point 的区间 [a,b] 端点函数值同号: f({a})={fa}, f({b})={fb}")
        # 在这种情况下，可能需要扩大搜索范围或检查函数行为
        # 这里我们返回一个标记，表示未找到
        return (a + b) / 2 # 作为一种妥协

    while b - a >= tolerance:
        c = (a + b) / 2
        fc = func(c, *args)
        if fc == 0:
            return c
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return (a + b) / 2

def position_iteration(theta_last, flag_last, segment_index):
    """计算单个铰接点的位置迭代"""
    d = 1.7
    D = 9
    theta0_const = 16.6319611
    r = 1.5027088
    aleph = 3.0214868
    
    if segment_index == 0:
        d0 = 3.41 - 0.275 * 2
        theta_1, theta_2, theta_3 = 0.9917636, 2.5168977, 14.1235657
    else:
        d0 = 2.2 - 0.275 * 2
        theta_1, theta_2, theta_3 = 0.5561483, 1.1623551, 13.8544471
    
    if flag_last == 1:
        def spiral_constraint(theta, d, d0, theta_last):
            return (theta**2 + theta_last**2 - 2*theta*theta_last*np.cos(theta-theta_last) - 4*np.pi**2*d0**2/d**2)
        theta = find_zero_point(spiral_constraint, theta_last, theta_last + np.pi, 1e-8, d, d0, theta_last)
        flag = 1
    elif flag_last == 2:
        if theta_last < theta_1:
            b = np.sqrt(2-2*np.cos(theta_last)) * r * 2
            beta = (aleph - theta_last) / 2
            l = np.sqrt(b**2 + D**2/4 - b*D*np.cos(beta))
            gamma = np.arcsin(b*np.sin(beta)/l)
            def transition_constraint(theta, d, d0, theta0, l, gamma):
                return (l**2 + d**2*theta**2/(4*np.pi**2) - d*l*theta*np.cos(theta-theta0+gamma)/np.pi - d0**2)
            theta = find_zero_point(transition_constraint, theta0_const, theta0_const + np.pi, 1e-8, d, d0, theta0_const, l, gamma)
            flag = 1
        else:
            theta = theta_last - theta_1
            flag = 2
    elif flag_last == 3:
        if theta_last < theta_2:
            a = np.sqrt(10-6*np.cos(theta_last)) * r
            phi = np.arccos((4*r**2+a**2-d0**2)/(4*a*r))
            beta = np.arcsin(r*np.sin(theta_last)/a)
            theta = aleph - phi + beta
            flag = 2
        else:
            theta = theta_last - theta_2
            flag = 3
    else: # flag_last == 4
        if theta_last < theta_3:
            p = d * (theta_last + np.pi) / (2 * np.pi)
            a = np.sqrt(p**2 + D**2/4 - p*D*np.cos(theta_last-theta0_const+np.pi))
            beta = np.arcsin(p*np.sin(theta_last-theta0_const+np.pi)/a)
            gamma = beta - (np.pi-aleph)/2
            b = np.sqrt(a**2 + r**2 - 2*a*r*np.cos(gamma))
            sigma = np.arcsin(a*np.sin(gamma)/b)
            phi = np.arccos((r**2+b**2-d0**2)/(2*r*b))
            theta = aleph - phi + sigma
            flag = 3
        else:
            def spiral_out_constraint(theta, d, d0, theta_last):
                t_val = theta + np.pi
                t_1 = theta_last + np.pi
                return (t_val**2 + t_1**2 - 2*t_val*t_1*np.cos(t_val-t_1) - 4*np.pi**2*d0**2/d**2)
            theta = find_zero_point(spiral_out_constraint, theta_last - np.pi, theta_last, 1e-8, d, d0, theta_last)
            flag = 4
    return theta, flag

# ============ 参数设置 ============ 
d = 1.7
v0 = 1
theta0 = 16.6319611
r = 1.5027088
aleph = 3.0214868
t1_time = 9.0808299
t2_time = 13.6212449
x1, y1 = -0.7600091, -1.3057264
x2, y2 = 1.7359325, 2.4484020
theta1_angle = 4.0055376
theta2_angle = 0.8639449
theta1 = 4.0055376
theta2 = 0.8639449
R = 4.5
LINK_WIDTH = 0.30

def get_positions_at_time(t):
    """实时计算t时刻所有铰接点的位置"""
    # 1. 计算龙头位置
    if t < 0:
        def head_equation(theta, theta0, v0, t, d):
            return (spiral_length_integral(theta0) - spiral_length_integral(theta) - 4*v0*t*np.pi/d)
        theta_head = find_zero_point(head_equation, 0, 100, 1e-8, theta0, v0, t, d)
        flag_head = 1
    elif t == 0:
        theta_head = theta0
        flag_head = 1
    elif t < t1_time:  # t1_time ≈ 9.08s, 在第一段圆弧
        theta_head = v0 * t / (2 * r)
        flag_head = 2
    elif t < t2_time:  # t2_time ≈ 13.62s, 在第二段圆弧
        theta_head = v0 * (t - t1_time) / r  # 注意这里是从t1_time开始计算，且半径是r不是2r
        flag_head = 3
    else:  # t >= t2_time, 在盘出螺线
        def head_equation_out(theta, theta0, v0, t_diff, d):
            return (spiral_length_integral(theta) - spiral_length_integral(theta0) - 4*v0*t_diff*np.pi/d)
        theta_head = find_zero_point(head_equation_out, theta0, theta0 + 50, 1e-8, theta0, v0, t - t2_time, d) - np.pi
        flag_head = 4
    
    # 2. 迭代计算所有铰接点位置参数
    thetas = [theta_head]
    flags = [flag_head]
    for i in range(223):
        theta_new, flag_new = position_iteration(thetas[-1], flags[-1], i)
        thetas.append(theta_new)
        flags.append(flag_new)
    
    # 3. 将位置参数转换为坐标
    coords = []
    for j in range(224):
        flag = flags[j]
        theta = thetas[j]
        if flag == 1:
            p = d * theta / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        elif flag == 2:
            x = x1 + 2 * r * np.cos(theta1_angle - theta)
            y = y1 + 2 * r * np.sin(theta1_angle - theta)
        elif flag == 3:
            x = x2 + r * np.cos(theta2_angle + theta - aleph)
            y = y2 + r * np.sin(theta2_angle + theta - aleph)
        else: # flag == 4
            p = d * (theta + np.pi) / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        coords.append((x, y))
    
    return coords

def draw_link_rectangle(ax, p1, p2, width, color='lightblue', alpha=0.7):
    """绘制连杆矩形"""
    direction = np.array(p2) - np.array(p1)
    length = np.linalg.norm(direction)
    if length < 1e-10: return
    unit_dir = direction / length
    perpendicular = np.array([-unit_dir[1], unit_dir[0]])
    half_width = width / 2
    corner1 = np.array(p1) - half_width * perpendicular
    corner2 = np.array(p1) + half_width * perpendicular
    corner3 = np.array(p2) + half_width * perpendicular
    corner4 = np.array(p2) - half_width * perpendicular
    rect_points = np.array([corner1, corner2, corner3, corner4, corner1])
    ax.plot(rect_points[:, 0], rect_points[:, 1], color='black', linewidth=0.5)
    ax.fill(rect_points[:-1, 0], rect_points[:-1, 1], color=color, alpha=alpha)

def draw_trajectory_lines(ax):
    """绘制运动轨迹线（使用正确的参数）"""
    # 螺线参数
    p0 = -8 * np.pi  # 盘入螺线起点极角，延伸到画面外
    p1 = theta1  # 盘入螺线终点极角（与第一段圆弧连接）
    p2 = theta2  # 盘出螺线起点极角（与第二段圆弧连接）
    p3 = theta2 + aleph + 8 * np.pi  # 盘出螺线终点极角，延伸到画面外
    
    # 1. 盘入螺线
    theta_in = np.linspace(2 * np.pi, 32 * np.pi, 2000)
    p_in = d * theta_in / (2 * np.pi)
    x_in = p_in * np.cos(theta_in)
    y_in = p_in * np.sin(theta_in)
    mask_in = np.sqrt(x_in**2 + y_in**2) >= 4.5
    ax.plot(x_in[mask_in], y_in[mask_in], color='green', linewidth=1, label='盘入螺线', zorder=10)

    # 2. 盘出螺线
    theta_out = np.linspace(0, 30 * np.pi, 2000)
    p_out = d * (theta_out + np.pi) / (2 * np.pi)
    x_out = p_out * np.cos(theta_out)
    y_out = p_out * np.sin(theta_out)
    mask_out = np.sqrt(x_out**2 + y_out**2) >= 4.5
    ax.plot(x_out[mask_out], y_out[mask_out], color='green', linewidth=1, label='盘出螺线', zorder=10)
    
    # 第一段圆弧
    arc1_theta = np.linspace(0, aleph, 300)
    arc1_x = x1 + 2 * r * np.cos(theta1- arc1_theta)
    arc1_y = y1 + 2 * r * np.sin(theta1 - arc1_theta)
    ax.plot(arc1_x, arc1_y, color='green', linewidth=1, label='第一段圆弧', zorder=10)
    
    # 第二段圆弧
    arc2_theta = np.linspace(0, aleph, 300)
    arc2_x = x2 + r * np.cos(theta2 + arc2_theta - aleph)
    arc2_y = y2 + r * np.sin(theta2 + arc2_theta - aleph)
    ax.plot(arc2_x, arc2_y, color='green', linewidth=1, label='第二段圆弧', zorder=10)
    
    # 3. 调头区域辅助圆
    circle = Circle((0, 0), R, color='#ffe599', alpha=0.5, zorder=-2)
    ax.add_patch(circle)
    # 圆边界
    ax.plot(R * np.cos(np.linspace(0, 2 * np.pi, 500)), R * np.sin(np.linspace(0, 2 * np.pi, 500)),
            color='#b7b7b7', linewidth=1, linestyle='--', label='调头区域边界', zorder=-1)

def plot_dragon_at_time(t, subplot_idx):
    """绘制t时刻的板凳龙状态"""
    coords = get_positions_at_time(t)
    
    plt.subplot(2, 3, subplot_idx)
    ax = plt.gca()
    
    draw_trajectory_lines(ax)
    
    for i in range(len(coords) - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        draw_link_rectangle(ax, p1, p2, LINK_WIDTH)
    
    for coord in coords:
        ax.plot(coord[0], coord[1], 'ko', markersize=2)
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f't = {t}s')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

# ============ 主程序：绘制多个时刻的图 ============ 
plt.figure(figsize=(18, 12))

# 设置要绘制的时间点：从t=0开始，步长25s
time_points = [0, 25, 50, 75, 100, 125]
for i, t in enumerate(time_points, 1):
    plot_dragon_at_time(t, i)

plt.suptitle('板凳龙运动状态可视化 (实时计算)', fontsize=16, y=0.95)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# 如果子图数量少于6，最后一个位置留给图例
if len(time_points) < 6:
    plt.subplot(2, 3, 6)
    plt.axis('off')
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=1.5, label='运动轨迹'),
        Rectangle((0, 0), 1, 1, facecolor='lightblue', alpha=0.7, label='板凳'),
        Line2D([0], [0], marker='o', color='black', markersize=4, linestyle='None', label='铰接点'),
        Rectangle((0, 0), 1, 1, facecolor='#ffe599', alpha=0.5, label='调头区域')
    ]
    plt.legend(handles=legend_elements, loc='center', fontsize=12)
plt.savefig('4plot2.png', dpi=300)
plt.show()
