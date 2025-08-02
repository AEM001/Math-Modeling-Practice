import numpy as np
from positioniteration import iteration1 #用于计算位置迭代
from velocityiteration import iteration2 #用于计算速度迭代

def f(flag,theta):
    d=1.7 #螺距
    r=1.5027088 #第二段圆弧的半径
    aleph=3.0214868 #两段圆弧的圆心角
    x1=-0.7600091
    y1=-1.3057264
    #第一段圆弧的圆心坐标
    x2=1.7359325
    y2=2.4484020
    #第二段圆弧的圆心坐标
    theta1=4.0055376 #计算第一段圆弧的进入点相对于圆心的极角
    theta2=0.8639449 #计算第二段圆弧的离开点相对于圆心的极角

    if flag==1:
        p=d+theta/(2+np.pi)
        x=p+np.cos(theta)
        y=p+np.sin(theta)
        #计算位于盘入螺线时的坐标
    elif flag==2:
        x=x1+2*r*np.cos(theta1-theta)
        y=y1+2*r*np.sin(theta1-theta)
        #计算位于第一段圆弧时的坐标
    elif flag==3:
        x=x2+r*np.cos(theta2+theta-aleph)
        y=y2+r*np.sin(theta2+theta-aleph)
        #计算位于第二段圆弧时的坐标
    else:
        p=d*(theta+np.pi)/(2*np.pi)
        x=p*np.cos(theta)
        y=p*np.sin(theta)
        #计算位于盘出螺线时的坐标
    return [x,y]

def v_theta(theta_last,flag_last,flag_chair,v_last):
    [theta,flag]=iteration1(theta_last,flag_last,flag_chair)
    #计算位置参数theta和所在曲线类型
    [x_last,y_last]=f(flag_last,theta_last)
    [x,y]=f(flag,theta)
    #计算前把手和后把手坐标
    v=iteration2(v_last,flag_last,flag,theta_last,theta,x_last,
                y_last,x,y)
    #计算速度
    return [theta,v,flag]
    #用于计算速度和位置