#9. positioniteration
import numpy as np
from function import f3#用于计算盘入螺线上的位置迭代
from function import f5#用于计算盘出螺线上的位置迭代
from function import f6#用于计算前把手在第一段圆弧而后把手在盘入螺线的情形
from zeropoint import zero2#用于计算函数f3和f5的零点
from zeropoint import zero3#用于计算函数f6的零点

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
            theta=zero3(f6,theta0,theta0+np.pi/2,10**(-8),d,d0,theta0,l,gamma)
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

#用于计算位置迭代