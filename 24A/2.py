# 问题二代码（整合所有依赖函数，去除import，便于独立运行）
import numpy as np
import pandas as pd

# ===== function.py 相关函数 =====
def f1(theta):
    result = theta*np.sqrt(theta**2+1) + np.log(theta+np.sqrt(theta**2+1))
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

# ===== zeropoint.py 相关函数 =====
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

# ===== number.py 相关函数 =====
def number(arr, decimals=6):
    if isinstance(arr, (list, np.ndarray)):
        for i in range(len(arr)):
            if isinstance(arr[i], (list, np.ndarray)):
                for j in range(len(arr[i])):
                    arr[i][j] = round(arr[i][j], decimals)
            else:
                arr[i] = round(arr[i], decimals)
    return arr

# ===== crashjudge.py 相关函数 =====
def judge(theta,d,v0):
    d1=0.275
    d2=0.15
    lst_theta=[theta]
    for i in np.arange(223):
        if i==0:
            d0=3.41-0.275 * 2
        else:
            d0=2.2-0.275 * 2
        theta_last=lst_theta[-1]
        a=theta_last
        b=theta_last+np.pi/2
        theta_new=zero2(f3,a,b,10**(-8),d,d0,theta_last)
        lst_theta.append(theta_new)
        if theta_new-theta>=3*np.pi:
            break
    lst_x=[]
    lst_y=[]
    for i in np.arange(len(lst_theta)):
        p=lst_theta[i]*d/(2*np.pi)
        x=p*np.cos(lst_theta[i])
        y=p*np.sin(lst_theta[i])
        lst_x.append(x)
        lst_y.append(y)
    lst_k=[]
    for i in np.arange(len(lst_theta)-1):
        k=(lst_y[i]-lst_y[i+1])/(lst_x[i]-lst_x[i+1])
        lst_k.append(k)
    k1=lst_k[0]
    x1=lst_x[0]
    y1=lst_y[0]
    k2=(d2/d1+k1)/(1-d2*k1/d1)
    b=d2*np.sqrt(k1**2+1)+y1-k1*x1
    if np.abs(b)<=np.abs(y1-k1*x1):
        b=-d2*np.sqrt(k1**2+1)+y1-k1*x1
    x=(y1-k2*x1-b)/(k1-k2)
    y=(k1*y1-k1*k2*x1-k2*b)/(k1-k2)
    flag=0
    for i in np.arange(len(lst_k)):
        if lst_theta[i+1]-theta>=np.pi:
            ki=lst_k[i]
            xi=lst_x[i]
            yi=lst_y[i]
            d_chair=np.abs(ki*(x-xi)+yi-y)/np.sqrt(ki**2+1)
            if d_chair<d2:
                flag=1
    return flag

# ======= 原主程序部分 =======
d=0.55#螺距
v0=1#龙头速度
theta0=32*np.pi#龙头初始极角
for theta in np.arange(60,0,-0.01):
    flag=judge(theta,d,v0)
    if flag:
        break
for theta in np.arange(theta+0.01,theta-0.01,-0.0001):
    flag=judge(theta,d,v0)
    if flag:
        break
for theta in np.arange(theta+0.0001,theta-0.0001,-0.000001):
    flag=judge(theta,d,v0)
    if flag:
        break
#细化碰撞时龙头的极角theta
theta_chair0=theta+0.000001
t=d*(f1(theta0)-f1(theta))/(4*np.pi*v0)#计算碰撞的时刻
lst_chair_theta=[theta_chair0]
for i in np.arange(223):
    if i==0:
        d0=3.41-0.275 * 2
    else:
        d0=2.2-0.275 * 2
    theta_last=lst_chair_theta[-1]#获得上一个把手的极角theta
    theta=zero2(f3,theta_last,theta_last+np.pi/2,10**(-8),d,d0,theta_last)
    lst_chair_theta.append(theta)
lst_chair_xyv=[]
for i in np.arange(224):
    lst_xyv=[]
    theta=lst_chair_theta[i]
    lst_xyv.append(d*theta*np.cos(theta)/(2*np.pi))
    lst_xyv.append(d*theta*np.sin(theta)/(2*np.pi))
    #根据theta角度确定把手坐标(x,y)
    lst_xyv.append(v0)
    lst_chair_xyv.append(lst_xyv)
lst_chair_xyv=np.array(lst_chair_xyv)
for i in np.arange(223):
    v_last=lst_chair_xyv[i,2] #获得上一个把手的速度
    theta_last=lst_chair_theta[i]
    theta=lst_chair_theta[i+1]
    x_last=lst_chair_xyv[i,0]
    y_last=lst_chair_xyv[i,1]
    x=lst_chair_xyv[i+1,0]
    y=lst_chair_xyv[i+1,1]
    #获得上一个把手和当前把手的坐标(x,y)和极角
    k_chair=(y_last-y)/(x_last-x)
    k_v_last=f4(theta_last)
    k_v=f4(theta)
    #计算板凳和两个速度的斜率
    aleph1=np.arctan(np.abs((k_v_last-k_chair)/(1+k_v_last*k_chair)))
    aleph2=np.arctan(np.abs((k_v-k_chair)/(1+k_v*k_chair)))
    #计算两个速度与板凳的夹角
    v=v_last*np.cos(aleph1)/np.cos(aleph2) #计算当前把手的速度
    lst_chair_xyv[i+1,2]=v
lst_chair_xyv=number(lst_chair_xyv,6) #保留6位小数
df=pd.DataFrame(lst_chair_xyv)
df.to_excel("result2_.xlsx",index=False)
#保存数据到Excel中

# ======= 新增输出部分 =======
print(f"终止时刻的时间 t = {t}")

index_list = [0, 50, 100, 150, 200]  # Python索引从0开始
for idx in index_list:
    x = lst_chair_xyv[idx, 0]
    y = lst_chair_xyv[idx, 1]
    v = lst_chair_xyv[idx, 2]
    print(f"第{idx+1}条龙身前把手: 位置=({x}, {y}), 速度={v}")

    x_tail = lst_chair_xyv[-(idx+1), 0]
    y_tail = lst_chair_xyv[-(idx+1), 1]
    v_tail = lst_chair_xyv[-(idx+1), 2]
    print(f"第{idx+1}条龙尾后把手: 位置=({x_tail}, {y_tail}), 速度={v_tail}")