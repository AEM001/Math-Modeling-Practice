import numpy as np
import pandas as pd

# ========== function.py ==========
def f1(theta):
    result = theta*np.sqrt(theta**2+1) + np.log(theta+np.sqrt(theta**2+1))
    return result

def f2(theta, theta0, v0, t, d):
    result = f1(theta0) - f1(theta) - 4*v0*t*np.pi/d
    return result

def f3(theta, d, d0, theta_last):
    t = theta
    t_1 = theta_last
    result = t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2
    return result

def f4(theta):
    t = theta
    result = (np.sin(t)+t*np.cos(t))/(np.cos(t)-t*np.sin(t))
    return result

def f5(theta, d, d0, theta_last):
    t = theta + np.pi
    t_1 = theta_last + np.pi
    result = t**2 + t_1**2 - 2*t*t_1*np.cos(t-t_1) - 4*np.pi**2*d0**2/d**2
    return result

def f6(theta, d, d0, theta0, l, gamma):
    t = theta
    t0 = theta0
    result = l**2 + d**2*t**2/(4*np.pi**2) - d*l*t*np.cos(t-t0+gamma)/np.pi - d0**2
    return result

# ========== zeropoint.py ==========
def zero1(f, a, b, eps, *args):
    while abs(b-a) > eps:
        c = (a+b)/2
        if f(c, *args) == 0:
            return c
        elif f(a, *args)*f(c, *args) < 0:
            b = c
        else:
            a = c
    return (a+b)/2

def zero2(f, a, b, eps, *args):
    while abs(b-a) > eps:
        c = (a+b)/2
        if f(c, *args) == 0:
            return c
        elif f(a, *args)*f(c, *args) < 0:
            b = c
        else:
            a = c
    return (a+b)/2

def zero3(f, a, b, eps, *args):
    while abs(b-a) > eps:
        c = (a+b)/2
        if f(c, *args) == 0:
            return c
        elif f(a, *args)*f(c, *args) < 0:
            b = c
        else:
            a = c
    return (a+b)/2

# ========== number.py ==========
def number(arr, decimals=6):
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr

# ========== 主体计算 ==========
d = 0.55  # 螺距
v0 = 1    # 龙头速度
theta0 = 32*np.pi  # 龙头初始极角

lst_chair_theta = []
for t in np.arange(301):
    if t == 0:
        theta_chair0 = theta0
    else:
        theta_chair0 = zero1(f2, 0, theta0, 10**(-8), theta0, v0, t, d)
    
    lst_theta = [theta_chair0]
    for i in np.arange(223):
        if i == 0:
            d0 = 3.41 - 0.275 * 2
        else:
            d0 = 2.2 - 0.275 * 2
        
        theta_last = lst_theta[-1]
        theta = zero2(f3, theta_last, theta_last+np.pi/2, 10**(-8), d, d0, theta_last)
        lst_theta.append(theta)
    
    lst_chair_theta.append(lst_theta)

lst_chair_theta = np.array(lst_chair_theta)

lst_chair_xy = []
for t in np.arange(301):
    lst_xy = []
    for i in np.arange(224):
        theta = lst_chair_theta[t, i]
        lst_xy.append(d*theta*np.cos(theta)/(2*np.pi))
        lst_xy.append(d*theta*np.sin(theta)/(2*np.pi))
    lst_chair_xy.append(lst_xy)

lst_chair_xy = np.array(lst_chair_xy).T
lst_chair_xy = number(lst_chair_xy, 6)

df = pd.DataFrame(lst_chair_xy)
df.to_excel("result1_1.xlsx", index=False)

lst_chair_v = []
for t in np.arange(0, 301):
    lst_v = [v0]
    for i in np.arange(223):
        v_last = lst_v[-1]
        theta_last = lst_chair_theta[t, i]
        theta = lst_chair_theta[t, i+1]
        x_last = lst_chair_xy[i*2, t]
        y_last = lst_chair_xy[i*2+1, t]
        x = lst_chair_xy[i*2+2, t]
        y = lst_chair_xy[i*2+3, t]
        
        k_chair = (y_last-y)/(x_last-x)
        k_v_last = f4(theta_last)
        k_v = f4(theta)
        
        aleph1 = np.arctan(np.abs((k_v_last-k_chair)/(1+k_v_last*k_chair)))
        aleph2 = np.arctan(np.abs((k_v-k_chair)/(1+k_v*k_chair)))
        v=v_last*np.cos(aleph1)/np.cos(aleph2)#计算当前把手的速度
        lst_v.append(v)
    lst_chair_v.append(lst_v)
lst_chair_v=np.array(lst_chair_v).T
lst_chair_v=number(lst_chair_v,6)#保留6位小数
df=pd.DataFrame(lst_chair_v)
df.to_excel("result1_2.xlsx",index=False)
#保存数据到Excel中

# ========== 输出指定时刻、指定节段把手的位置和速度 ==========
# 指定时刻
time_points = [0, 60, 120, 180, 240, 300]
# 指定节段索引
seg_idx = [0, 1, 51, 101, 151, 201, 223]  # 223为龙尾后把手

print("时刻(s)\t节段\t位置(x, y)\t\t速度")
for t in time_points:
    for idx in seg_idx:
        x = lst_chair_xy[idx*2, t]
        y = lst_chair_xy[idx*2+1, t]
        v = lst_chair_v[idx, t]
        print(f"{t}\t{idx}\t({x}, {y})\t{v}")