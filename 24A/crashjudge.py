import numpy as np
from function import f3#用于计算盘入螺线上的位置迭代
from zeropoint import zero2#用于计算函数f3的零点
def judge(theta,d,v0):
    d1=0.275
    d2=0.15
    lst_theta=[theta]
    for i in np.arange(223):
        if i==0:
            d0=3.41-0.275 * 2
        else:
            d0=2.2-0.275 * 2
        #确定板长
        theta_last=lst_theta[-1]#获得上一个把手的极角theta
        a=theta_last
        b=theta_last+np.pi/2
        theta_new=zero2(f3,a,b,10**(-8),d,d0,theta_last)
        lst_theta.append(theta_new)
        #计算当前把手的极角theta
        if theta_new-theta>=3*np.pi:
            break
        #当当前把手位置离龙头过远时结束
    lst_x=[]
    lst_y=[]
    for i in np.arange(len(lst_theta)):
        p=lst_theta[i]*d/(2*np.pi)
        x=p*np.cos(lst_theta[i])
        y=p*np.sin(lst_theta[i])
        lst_x.append(x)
        lst_y.append(y)
    #根据theta角度确定把手坐标(x,y)
    lst_k=[]
    for i in np.arange(len(lst_theta)-1):
        k=(lst_y[i]-lst_y[i+1])/(lst_x[i]-lst_x[i+1])
        lst_k.append(k)
    #计算板凳的斜率
    k1=lst_k[0]
    x1=lst_x[0]
    y1=lst_y[0]
    #获得龙头的坐标和斜率
    k2=(d2/d1+k1)/(1-d2*k1/d1)#计算龙头前把手和外前点直线的斜率
    b=d2*np.sqrt(k1**2+1)+y1-k1*x1
    if np.abs(b)<=np.abs(y1-k1*x1):
        b=-d2*np.sqrt(k1**2+1)+y1-k1*x1
    #计算龙头外侧边的截距
    x=(y1-k2*x1-b)/(k1-k2)
    y=(k1*y1-k1*k2*x1-k2*b)/(k1-k2)
    #计算龙头外前点的坐标

    flag=0

    for i in np.arange(len(lst_k)):
        if lst_theta[i+1]-theta>=np.pi:
            ki=lst_k[i]
            xi=lst_x[i]
            yi=lst_y[i]
            d_chair=np.abs(ki*(x-xi)+yi-y)/np.sqrt(ki**2+1)
            #计算龙头外前点到当前板凳中心线的距离
            if d_chair<d2:
                flag=1
                #判断是否相撞

    x2=lst_x[1]
    y2=lst_y[1]
    #获得第一节龙身前把手的坐标

    k2=(k1-d2/d1)/(1+d2*k1/d1) #计算第一节龙身前把手和龙头外后点直线的斜率
    x=(y2-k2*x2-b)/(k1-k2)
    y=(k1*y2-k1*k2*x2-b*k2)/(k1-k2)
    #计算龙头外后点的坐标

    for i in np.arange(len(lst_k)):
        if lst_theta[i+1]-theta>=np.pi:
            ki=lst_k[i]
            xi=lst_x[i]
            yi=lst_y[i]
            d_chair=np.abs(ki*(x-xi)+yi-y)/np.sqrt(ki**2+1)
            #计算龙头外后点到当前板凳中心线的距离
            if d_chair<d2:
                flag=1
                #判断是否相撞

    k1=lst_k[1]
    x1=lst_x[1]
    y1=lst_y[1]
    #获得第一节龙身的前把手坐标和斜率

    k2=(d2/d1+k1)/(1-d2*k1/d1) #计算第一节龙身前把手和外前点直线的斜率
    b=d2*np.sqrt(k1**2+1)+y1-k1*x1
    if np.abs(b)<=np.abs(y1-k1*x1):
        b=-d2*np.sqrt(k1**2+1)+y1-k1*x1
    #计算第一节龙身外侧边的截距

    x=(y1-k2*x1-b)/(k1-k2)
    y=(k1*y1-k1*k2*x1-k2*b)/(k1-k2)
    #计算第一节龙身外前点的坐标

    for i in np.arange(len(lst_k)):
        if lst_theta[i+1]-theta>=np.pi:
            ki=lst_k[i]
            xi=lst_x[i]
            yi=lst_y[i]
            d_chair=np.abs(ki*(x-xi)+yi-y)/np.sqrt(ki**2+1)
            #计算第一节龙身外前点到当前板凳中心线的距离
            if d_chair<d2:
                flag=1
                #判断是否相撞

    x2=lst_x[2]
    y2=lst_y[2]
    #获得第二节龙身前把手的坐标

    k2=(k1-d2/d1)/(1+d2*k1/d1) #计算第二节龙身前把手和第一节龙身外后点直线的斜率
    x=(y2-k2*x2-b)/(k1-k2)
    y=(k1*y2-k1*k2*x2-b*k2)/(k1-k2)
    #计算第一节龙身外后点的坐标

    for i in np.arange(len(lst_k)):
    
        if lst_theta[i+1]-theta>=np.pi:
        
            ki=lst_k[i]
        
            xi=lst_x[i]
        
            yi=lst_y[i]
        
            d_chair=np.abs(ki*(x-xi)+yi-y)/np.sqrt(ki**2+1)
        
        #计算第一节龙身外后点到当前板凳中心线的距离
        
            if d_chair<d2:
            
                flag=1
            
            #判断是否相撞
            
    return flag
    
    #用于判断是否发生碰撞