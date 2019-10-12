# Import our modules that we are using
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import special as sp
from sympy import *
from bspline import Bspline
from functools import partial

def getBezierCurveBasisFunctions(n):
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-t)**(n-i))*(t**i)) for i in range(n+1) ]


def getBezierSurfaceBasisFunctions(n):
    s = symbols('s')
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-s)**(n-i))*(s**i)*int(sp.comb(n,j))*((1-t)**(n-j))*(t**j)) for i in range(n+1) for j in range(n+1) ]


knot_vector = [1,2,3,4,5]


def plotBsplineBasisFunctions(knot_vector,degree):
    basis = Bspline(knot_vector,degree)
    basis.plot()
    
#plotBsplineBasisFunctions(knot_vector,1)         
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
            #break;
        ax.legend()
        plt.title("Bezier surfaces for degree " + str(degree))
        plt.xlabel('t')
        plt.ylabel('s')
        plt.show()
        
#plotBezierBasisFunctions(type='curve',degree=1)

#plotBezierBasisFunctions(type='surface',degree=1)

def plotBezierCurves(controlPoints,degree):
    if (len(controlPoints) != degree +1):
        raise ValueError('The number of control points should be 1 more than the degree')
    controlPoints = np.array(controlPoints)
    cpX , cpY = controlPoints[:,0] , controlPoints[:,1]
    t = np.arange(0,1+0.01,0.01)
    basisFunctions = getBezierCurveBasisFunctions(degree)
    curve, i = 0,0
    for func in basisFunctions:
        curve += np.outer(eval(func, {'t':t}),controlPoints[i])
        i += 1
    curveX , curveY = curve[:,0] , curve[:,1]
    plt.plot(curveX,curveY, label = "Bezier Curve")
    plt.plot(cpX,cpY,'-og' , label = "Control Polygon")
    for i in range(len(cpX)):
        plt.annotate("P"+str(i), (cpX[i],cpY[i]), textcoords="offset points", xytext=(0,10), ha='center') 
    plt.title("Bezier curves for degree " + str(degree))
    plt.xlabel('t')
    plt.ylabel('y axis')
    plt.xlim([min(cpX) - 0.3, max(cpX) + 0.3])
    plt.ylim([min(cpY) - 0.3, max(cpY) + 0.3])
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()

d = {2: [[0,0],[0.5,1],[1,0]], 1:[[0,0],[0.5,1]],3:[[0,0],[0.5,1],[0.5,0],[1,1]]}   
plotBezierCurves(d[3],3)    
