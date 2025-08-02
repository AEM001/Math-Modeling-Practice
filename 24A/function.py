import numpy as np

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