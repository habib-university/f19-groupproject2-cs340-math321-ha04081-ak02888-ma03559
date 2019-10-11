# Import our modules that we are using
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import special as sp
from sympy import *

def getBezierCurveBasisFunctions(n):
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-t)**(n-i))*(t**i)) for i in range(n+1) ]


def getBezierSurfaceBasisFunctions(n):
    s = symbols('s')
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-s)**(n-i))*(s**i)*int(sp.comb(n,j))*((1-t)**(n-j))*(t**j)) for i in range(n+1) for j in range(n+1) ]
    
def plotBezierBasisFunctions(type='curve',degree=1):
    t = np.arange(0,1+0.01,0.01)
    if (type == 'curve'):
        basisFunctions = getBezierCurveBasisFunctions(degree)
        for func in basisFunctions:
            plt.plot(t,eval(func, {'t':t}),label = func)
        plt.title("Bezier curves for degree " + str(degree))
        plt.xlabel('t')
        plt.ylabel('y axis')
        plt.grid(alpha=.4,linestyle='--')
        plt.legend()
        plt.show()
    if (type == 'surface'):
        s = np.arange(0,1+0.01,0.01)
        T, S = np.meshgrid(t, s)
        fig = plt.figure()
        ax = Axes3D(fig)
        basisFunctions = getBezierSurfaceBasisFunctions(degree)
        for func in basisFunctions:
            c = ax.plot_surface(T, S, eval(func,{'t':T,'s':S}),label=func)
            c._facecolors2d=c._facecolors3d
            c._edgecolors2d=c._edgecolors3d
        ax.legend()
        plt.title("Bezier surfaces for degree " + str(degree))
        plt.xlabel('t')
        plt.ylabel('s')
        plt.show()
        
#plotBezierBasisFunctions(type='curve',degree=2)

plotBezierBasisFunctions(type='surface',degree=2)

        
    
    
