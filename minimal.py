import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Minimal():
    def __init__(self, lim=None):
        self.F = self.six_hump_camel_back
        if lim == None:
            self.lim = np.array([[-10,-10],[10,10]])
        else:
            self.lim = lim
    
    def af(self,a):
        x = a[0]
        y = a[1]
        return - 3*(1-x)**2*np.exp(-(x**2)-(y+1)**2) + 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) + 1/3**np.exp(-(x+1)**2 - y**2)

    def Rana(self,a):
        sum = 0
        for i in range(len(a)-1):
            p1 = a[i]
            p2 = a[i+1]
            sum += (p1*np.sin(np.sqrt(np.fabs(p2+1-p1)))*np.cos(np.sqrt(np.fabs(p1+p2+1)))+
                    (p2+1)*np.cos(np.sqrt(np.fabs(p2+1-p1)))*np.sin(np.sqrt(np.fabs(p1+p2+1))))
        return sum
    
    # global optimum x* = (0.08983,0.7126) and f(x*) = -1.0316285 for 5 < x(i) < 5
    def six_hump_camel_back(self,a):
        return 4*a[0]**2 -2.1*a[0]**4+1/3*a[0]**6+a[0]*a[1]-4*a[1]**2+4*a[1]**4


    def sample(self):
        return np.random.rand(2)*(self.lim[1] - self.lim[0])+self.lim[0]
    
    def drawResult(self,p,v):        
        discrete_num = 50
        x = np.linspace(self.lim[0][0], self.lim[1][0], discrete_num) 
        y = np.linspace(self.lim[0][1], self.lim[1][1], discrete_num) 
        x,y = np.meshgrid(x,y)
        np.meshgrid()

        z = self.Rana([x,y])
        
        plt.contour(x,y,z)
        plt.plot(p[0],p[1],'x')
        plt.text(self.lim[0][0]+1,self.lim[0][1]+1,r'$f(%.4f,%.4f)=%.4f$'%(p[0],p[1],v),fontdict={'size':'9','color':'b'})

        # fig = plt.figure()
        # ax = Axes3D(fig) 
        # ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
        # ax.plot(p[0],p[1],v,'.')

"""
discrete_num = 50
x = np.linspace(xlim[0], xlim[1], discrete_num) 
y = np.linspace(ylim[0], ylim[1], discrete_num) 
x,y = np.meshgrid(x,y)
z = F(x,y)

fig = plt.figure()
ax = Axes3D(fig) 
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')
plt.show()
"""

class SA():
    def __init__(self, prob_v, prob_p, prob_s=None, T=100, T_end=1e-3, L=100, r=0.9):
        self.T = T              
        self.T_end = T_end    
        self.L = L              
        self.r = r    

        self.f = prob_v 
        self.sample = prob_p

        self.x = self.sample()
        self.v = self.f(self.x)

        self.iter_num = 0
        # print
        self.vm = []
        self.tm = []
        if prob_s == None:
            self.drawX = print
        else:
            self.drawX = prob_s

    # def new_sample(self):
    #     return np.random.rand()*(self.lim[1] - self.lim[0])
    
    """accept new"""
    def metropolis(self, n):
        nv = self.f(n)
        dv = nv - self.v 
        if dv < 0 or np.exp(-dv/self.T) > np.random.rand():
            self.x = n
            self.v = nv

        
    def solve(self):
        while self.T > self.T_end:
            for i in range(self.L):
                n = self.sample()
                self.metropolis(n)
                self.vm.append(self.v)
            self.T *= self.r
            self.iter_num += 1
            self.tm.append(self.T)
    
    def drawValueChange(self):
        plt.subplot(1,2,1)
        self.drawX(self.x, self.v)
        print("minimal value: ",self.v)
        plt.subplot(1,2,2)
        plt.plot(list(range(self.iter_num*self.L)),self.vm)
        #plt.plot(np.linspace(0,self.iter_num*self.L,self.iter_num),self.tm,'r')
        plt.show()

m = Minimal()
sa = SA(m.F, m.sample, m.drawResult)
sa.solve()
sa.drawValueChange()

BENCH = False
if BENCH:
    bench = []
    for i in range(100):
        sa = SA(m.F, m.sample)
        sa.solve()
        bench.append([sa.x,sa.v])
    bench = np.array(bench)
    np.save('six_h_-3_2_100',bench)