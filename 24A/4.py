import numpy as np
import pandas as pd

# 从function.py整合的函数
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

# 从zeropoint.py整合的函数
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

# 从number.py整合的函数
def number(arr, decimals=6):
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr

# 从positioniteration.py整合的函数
def iteration1(theta_last,flag_last,flag_chair):
    d=1.7#螺距
    D=9#调头空间的直径
    theta0=16.6319611#龙头0时刻时的极角
    r=1.5027088#第二段圆弧的半径
    aleph=3.0214868#两段圆弧的圆心角
    
    if flag_chair==0:
        d0=3.41-0.275 * 2
        theta_1=0.9917636
        theta_2=2.5168977
        theta_3=14.1235657
    else:
        d0=2.2-0.275 * 2
        theta_1=0.5561483
        theta_2=1.1623551
        theta_3=13.8544471
    #确定板长和三个重要位置参数
    
    if flag_last==1:
        theta=zero2(f3,theta_last,theta_last+np.pi/2,10**(-8),d,d0,
                    theta_last)
        #计算后把手的位置参数theta
        flag=1#返回后把手所在曲线的类型
        #计算前把手和后把手都在盘入螺线的情形
    elif flag_last==2:
        if theta_last<theta_1:
            b=np.sqrt(2-2*np.cos(theta_last))*r*2
            beta=(aleph-theta_last)/2
            l=np.sqrt(b**2+D**2/4-b*D*np.cos(beta))
            gamma=np.arcsin(b*np.sin(beta)/l)
            theta=zero3(f6,theta0,theta0+np.pi/2,10**(-8),d,d0,theta0,1,gamma)
            flag=1#返回后把手所在曲线的类型
            
        #计算后把手的位置参数theta
            
        #计算前把手在第一段圆弧而后把手在盘入螺线的情形
        else:
            theta=theta_last-theta_1
            #计算后把手的位置参数theta
            flag=2#返回后把手所在曲线的类型
            #计算前把手和后把手都在第一段圆弧的情形
    elif flag_last==3:
        if theta_last<theta_2:
            a=np.sqrt(10-6*np.cos(theta_last))*r
            phi=np.arccos((4*r**2+a**2-d0**2)/(4*a*r))
            beta=np.arcsin(r*np.sin(theta_last)/a)
            theta=aleph-phi+beta
            #计算后把手的位置参数theta
            flag=2#返回后把手所在曲线的类型
            #计算前把手在第二段圆弧而后把手在第一段圆弧的情形
        else:
            theta=theta_last-theta_2
            #计算后把手的位置参数theta
            flag=3#返回后把手所在曲线的类型
            #计算前把手和后把手都在第二段圆弧的情形
    else:
        if theta_last<theta_3:
            p=d*(theta_last+np.pi)/(2*np.pi)
            a=np.sqrt(p**2+D**2/4-p*D*np.cos(theta_last-theta0+np.pi))
            beta=np.arcsin(p*np.sin(theta_last-theta0+np.pi)/a)
            gamma=beta-(np.pi-aleph)/2
            b=np.sqrt(a**2+r**2-2*a*r*np.cos(gamma))
            sigma=np.arcsin(a*np.sin(gamma)/b)
            phi=np.arccos((r**2+b**2-d0**2)/(2*r*b))
            theta=aleph-phi+sigma
            #计算后把手的位置参数theta
            flag=3#返回后把手所在曲线的类型
            #计算前把手在盘出螺线而后把手在第二段圆弧的情形
        else:
            a=theta_last-np.pi/2
            b=theta_last
            theta=zero2(f5,a,b,10**(-8),d,d0,theta_last)
            #计算后把手的位置参数theta
            flag=4#返回后把手所在曲线的类型
            #计算前把手和后把手都在盘出螺线的情形
    return [theta,flag]

# 从velocityiteration.py整合的函数
def iteration2(v_last,flag_last,flag,theta_last,theta,x_last,y_last,x,y):
    x1=-0.7600091
    y1=-1.3057264
    # 计算第一段圆弧的圆心坐标
    x2=1.7359325
    y2=2.4484020
    # 计算第二段圆弧的圆心坐标
    k_chair=(y_last - y)/(x_last - x)  # 计算板凳的斜率
    v=-1
    if flag_last==1 and flag==1:
        k_v_last=f4(theta_last)
        k_v=f4(theta)
        # 计算前把手和后把手都在盘入螺线时两个速度的斜率
    elif flag_last==2 and flag==1:
        k_v_last=-(x_last - x1)/(y_last - y1)
        k_v=f4(theta)
        # 计算前把手在第一段圆弧而后把手在盘入螺线时两个速度的斜率
    elif flag_last==2 and flag==2:
        v=v_last
        # 计算前把手和后把手都在第一段圆弧的情形
    elif flag_last==3 and flag==2:
        k_v_last=-(x_last - x2)/(y_last - y2)
        k_v=-(x - x1)/(y - y1)
        # 计算前把手在第二段圆弧而后把手在第一段圆弧时两个速度的斜率
    elif flag_last==3 and flag==3:
        v=v_last
        # 计算前把手和后把手都在第二段圆弧的情形
    elif flag_last==4 and flag==3:
        theta_last=theta_last+np.pi
        k_v_last=f4(theta_last)
        k_v=-(x - x2)/(y - y2)
        # 计算前把手在盘出螺线而后把手在第二段圆弧时两个速度的斜率
    else:
        theta_last=theta_last+np.pi
        theta=theta - np.pi
        k_v_last=f4(theta_last)
        k_v=f4(theta)
        # 计算前把手和后把手都在盘出螺线时两个速度的斜率
    if v==-1:
        alph1=np.arctan(np.abs((k_v_last - k_chair)/(1 + k_v_last * k_chair)))
        alph2=np.arctan(np.abs((k_v - k_chair)/(1 + k_v * k_chair)))
        # 计算两个速度与板凳的夹角
        v=v_last * np.cos(alph1)/np.cos(alph2)  # 计算当前把手的速度
    return v

d = 1.7  # 螺距
v0 = 1  # 龙头速度
theta0 = 16.6319611  # 龙头0时刻时的极角
r = 1.5027088  # 第二段圆弧的半径
aleph = 3.0214868  # 两段圆弧的圆心角
t1 = 9.0808299  # 龙头到达第二段圆弧的时刻
t2 = 13.6212449  # 龙头到达盘出螺线的时刻

x1 = -0.7600091
y1 = -1.3057264
# 第一段圆弧的圆心坐标

x2 = 1.7359325
y2 = 2.4484020
# 第二段圆弧的圆心坐标

theta1 = 4.0055376  # 第一段圆弧的进入点相对于圆心的极角
theta2 = 0.8639449  # 第二段圆弧的离开点相对于圆心的极角

lst_chair_theta = []
lst_chair_flag = []
# 接续前文代码...

# 计算龙头和把手的位置参数及曲线类型
for t in np.arange(-100, 101):
    if t < 0:
        theta_chair0 = zero1(f2, theta0, 100, 10**(-8), theta0, v0, t, d)
        flag_chair0 = 1
    elif t == 0:
        theta_chair0 = theta0
        flag_chair0 = 1
    elif t < t1:
        theta_chair0 = v0 * t / (2 * r)
        flag_chair0 = 2
    elif t < t2:
        theta_chair0 = v0 * (t - t1) / r
        flag_chair0 = 3
    else:
        theta_chair0 = zero1(f2, theta0, 100, 10**(-8), theta0, v0, -t + t2, d)
        flag_chair0 = 4
    
    # 存储龙头的位置参数theta和所在曲线的类型参数flag
    lst_theta = [theta_chair0]
    lst_flag = [flag_chair0]
    
    # 计算每个把手的位置参数
    for i in np.arange(223):
        theta_last = lst_theta[-1]  # 上一个把手的theta
        flag_last = lst_flag[-1]    # 上一个把手的flag
        theta, flag = iteration1(theta_last, flag_last, i)
        lst_theta.append(theta)
        lst_flag.append(flag)
    
    lst_chair_theta.append(lst_theta)
    lst_chair_flag.append(lst_flag)

# 转换为NumPy数组
lst_chair_flag = np.array(lst_chair_flag)
lst_chair_theta = np.array(lst_chair_theta)

# 计算每个把手的坐标(x,y)
lst_chair_xy = []
for i in np.arange(201):
    lst = []
    for j in np.arange(224):
        flag = lst_chair_flag[i, j]  # 当前把手的flag
        theta = lst_chair_theta[i, j]  # 当前把手的theta
        
        if flag == 1:
            p = d * theta / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        elif flag == 2:
            x = x1 + 2 * r * np.cos(theta1 - theta)
            y = y1 + 2 * r * np.sin(theta1 - theta)
        elif flag == 3:
            x = x2 + r * np.cos(theta2 + theta - aleph)
            y = y2 + r * np.sin(theta2 + theta - aleph)
        else:
            p = d * (theta + np.pi) / (2 * np.pi)
            x = p * np.cos(theta)
            y = p * np.sin(theta)
        
        lst.append(x)
        lst.append(y)
    lst_chair_xy.append(lst)

# 整理数据并创建DataFrame
lst_chair_xy = np.array(lst_chair_xy).T
lst_chair_xy = number(lst_chair_xy, 6)  # 保留6位小数
df = pd.DataFrame(lst_chair_xy)
# 接续前文所有代码...

# 将坐标数据保存到第一个Excel文件
df.to_excel("result4_1.xlsx", index=False)

# 计算并保存速度数据
lst_chair_v = [] 
for i in np.arange(201):
    lst_v = [v0]  # 初始化第一个把手的速度为v0
    for j in np.arange(223):
        # 获取上一个把手和当前把手的参数
        flag_last = lst_chair_flag[i, j]
        theta_last = lst_chair_theta[i, j]
        flag = lst_chair_flag[i, j+1]
        theta = lst_chair_theta[i, j+1]
        x_last = lst_chair_xy[j*2, i]
        y_last = lst_chair_xy[j*2+1, i]
        x = lst_chair_xy[j*2+2, i]
        y = lst_chair_xy[j*2+3, i]
        
        # 获得上一个把手的速度
        v_last = lst_v[-1]
        
        # 计算当前把手的速度
        v = iteration2(v_last, flag_last, flag, theta_last, theta, 
                      x_last, y_last, x, y)
        lst_v.append(v)
    
    lst_chair_v.append(lst_v)

# 处理速度数据并保存到第二个Excel文件
lst_chair_v = np.array(lst_chair_v).T
lst_chair_v = number(lst_chair_v, 6)  # 保留6位小数
df = pd.DataFrame(lst_chair_v)
df.to_excel("result4_2.xlsx", index=False)