def zero1(f,a,b,e,theta0,v0,t,d):
    while b-a>=e:
        c=(a+b)/2 #取中点
        if f(a,theta0,v0,t,d)*f(c,theta0,v0,t,d)<0:
            b=c
        else:
            a=c
        #判断零点所在区间
    return (a+b)/2
#用于计算函数f2的零点
def zero2(f,a,b,e,d,d0,theta_last):
    while b-a>=e:
        c=(a+b)/2 #取中点
        if f(a,d,d0,theta_last)*f(c,d,d0,theta_last)<0:
            b=c
        else:
            a=c
        #判断零点所在区间
    return (a+b)/2
#用于计算函数f3和f5的零点
def zero3(f,a,b,e,d,d0,theta0,l,gamma):
    while b-a>=e:
        c=(a+b)/2 #取中点
        if f(a,d,d0,theta0,l,gamma)*f(c,d,d0,theta0,l,gamma)<0:
            b=c
        else:
            a=c
        #判断零点所在区间
    return (a+b)/2
    #用于计算函数f6的零点