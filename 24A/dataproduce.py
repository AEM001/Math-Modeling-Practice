## 8. dataproduce
import numpy as np
from function import f4#用于计算螺线上速度的斜率
from function import f5#用于计算盘出螺线上的位置迭代
from zeropoint import zero2#用于计算函数f5的零点

d=1.7#螺距
v0=1#龙头速度
D=9#调头空间的直径
d0_1=3.41-0.275 * 2#龙头板长
d0_2=2.2-0.275 * 2#龙身和龙尾板长

theta0=D*np.pi/d#计算龙头0时刻时的极角
x0=D*np.cos(theta0)/2
y0=D*np.sin(theta0)/2
#计算龙头0时刻的坐标(x,y)

k=f4(theta0)
11=2*np.abs(y0-k*x0)/np.sqrt(k**2+1)
12=np.sqrt(D**2-11**2)
r=D**2/(6 * 11)#计算第二段圆弧的半径

x1=x0+2*r/np.sqrt(1+1/k**2)
y1=-(x1-x0)/k+y0
#计算第一段圆弧的圆心坐标

x2=-x0-r/np.sqrt(1+1/k**2)
y2=-(x2+x0)/k-y0
#计算第二段圆弧的圆心坐标

aleph=np.arccos(12/(r*3))+np.pi/2#计算两段圆弧的圆心角

theta_chair0_1=np.arccos((8*r**2-d0_1**2)/(8*r**2))
#计算第一节龙身前把手到达第一段圆弧时龙头的位置参数theta

theta_chair0_2=np.arccos((2*r**2-d0_1**2)/(2*r**2))
#计算第一节龙身前把手到达第二段圆弧时龙头的位置参数theta

a=theta0-np.pi
b=theta0-np.pi/2
theta_chair0_3=zero2(f5,a,b,10**(-8),d,d0_1,theta0-np.pi)
#计算第一节龙身前把手到达盘出螺线时龙头的位置参数theta

theta_chair_1=np.arccos((8*r**2-d0_2**2)/(8*r**2))
#计算龙身后把手到达第一段圆弧时前把手的位置参数theta
theta_chair_2=np.arccos((2*r**2-d0_2**2)/(2*r**2))

#计算龙身后把手到达第二段圆弧时前把手的位置参数theta
theta_chair_3=zero2(f5,a,b,10**(-8),d,d0_2,theta0-np.pi)

#计算龙身后把手到达盘出螺线时前把手的位置参数theta
t1=2*r*aleph/v0#计算龙头到达第二段圆弧的时刻
t2=t1+r*aleph/v0#计算龙头到达盘出螺线的时刻

theta1=np.arctan((y1-y0)/(x1-x0))+np.pi#计算第一段圆弧的进入点相对于圆心的极角
theta2=np.arctan((y2+y0)/(x2+x0))#计算第二段圆弧的离开点相对于圆心的极角
x_=-x0/3
y_=-y0/3
#计算两段圆弧交界点的坐标

print(theta0,x0,y0,r,x1,y1,x2,y2,aleph,theta_chair0_1,
      theta_chair0_3,theta_chair_1,theta_chair_2,theta_chair_3,t1,
      theta2,x_,y_)
#输出各项重要参数