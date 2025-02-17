# Import our modules that we are using
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import numpy as np
from scipy import special as sp
from sympy import *
from bspline import Bspline
from matplotlib import cm
from scipy.interpolate import griddata


def getBezierCurveBasisFunctions(n):
    """
    Paramters:
        n : degree of curve
            n = 1 for linear, n = 2 for quadratic and so on
    Returns:
        List of basis functions for bezier curve for the given n
    Example:
        >>> getBezierCurveBasisFunctions(1)
            ['-t + 1', 't']
    """
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-t)**(n-i))*(t**i)) for i in range(n+1) ]


def getBezierSurfaceBasisFunctions(n):
    """
    Paramters:
        n : degree of surface
            n = 1 for linear, n = 2 for quadratic and so on
    Returns:
        List of basis functions for bezier surface for the given n
    Example:
        >>> getBezierSurfaceBasisFunctions(1)
            ['(-s + 1)*(-t + 1)', 't*(-s + 1)', 's*(-t + 1)', 's*t']
    """
    s = symbols('s')
    t = symbols('t')
    return [str(int(sp.comb(n,i))*((1-s)**(n-i))*(s**i)*int(sp.comb(n,j))*((1-t)**(n-j))*(t**j)) for i in range(n+1) for j in range(n+1) ]

'''#BSpline using library
def plotBsplineBasisFunctions(knot_vector,degree):
    basis = Bspline(knot_vector,degree)
    basis.plot()
'''

def N_ik(knot,i,k,t):
    """
    Paramters:
        knot : the knot vector of discrete values
        i : this refers to the index of basis functions --> N_i,k
        k : order of basis functions; degree = k - 1
        t : interpolated vector of knot vector 
    Returns:
        N_i,k for a given knot vector
    """
    if (k == 1):
        #base condition
        return np.array(list(map(int,(knot[i] <= t) & (knot[i+1] > t))))
    else:
        #checking if denominators are zero to avoid division by zero error
        term1 = (t - knot[i])/(knot[i+k-1]-knot[i])*N_ik(knot,i,k-1,t) if (knot[i+k-1]-knot[i]) != 0 else np.zeros(shape=(t.shape))
        term2 = (knot[i+k] - t)/(knot[i+k]-knot[i+1])*N_ik(knot,i+1,k-1,t) if (knot[i+k]-knot[i+1]) != 0 else np.zeros(shape=(t.shape))
        return term1 + term2  

def bSplineBasis(knot, k,t):
    """
    Paramters:
        knot : the knot vector of discrete values
        k : order of basis functions; degree = k - 1
        t : interpolated vector of knot vector 
    Returns:
        A list of all N_i,k for a given knot vector interpolated over t
    """
    return [N_ik(knot,i,k,t) for i in range(len(knot)-k)]
            
def plotBsplineBasisFunctions(knot, k):
    """
    Paramters:
        knot : the knot vector of discrete values
        k : order of basis functions; degree = k - 1 
    Output:
        Plots the B-spline basis functions
    """
    t = np.linspace(knot[0],knot[-1],1000)
    funcs = bSplineBasis(knot, k,t)
    i = 0
    for func in  funcs:
        plt.plot(t,func,label = "N_" + str(i) + "," + str(k))
        i += 1
    plt.title("BSpline basis functions for k =  " + str(k) + " and t = " + str(knot))
    plt.xlabel('t')
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()
    
def plotBezierBasisFunctions(type='curve',degree=1):
    """
    Paramters:
        type : curve or surface
        degree : degree of bezier basis functions, degree = 1 for linear and so on
    Output:
        Plots the bezier basis functions for curves if type is curve
        Plots the bezier basis functions for patches if type is surface
    """
    t = np.linspace(0,1,1000)
    if (type == 'curve'):
        basisFunctions = getBezierCurveBasisFunctions(degree)
        for func in basisFunctions:
            plt.plot(t,eval(func, {'t':t}),label = func)
        plt.title("Bezier basis functions over unit interval for degree " + str(degree))
        plt.xlabel('t')
        plt.grid(alpha=.4,linestyle='--')
        plt.legend()
        plt.show()
    if (type == 'surface'):
        s = np.linspace(0,1,1000)
        T, S = np.meshgrid(t, s)
        fig = plt.figure()
        ax = Axes3D(fig)
        basisFunctions = getBezierSurfaceBasisFunctions(degree)
        for func in basisFunctions:
            c = ax.plot_surface(T, S, eval(func,{'t':T,'s':S}),label=func)
            c._facecolors2d=c._facecolors3d
            c._edgecolors2d=c._edgecolors3d
        ax.legend()
        plt.title("Bezier basis functions over unit square for degree " + str(degree))
        plt.xlabel('t')
        plt.ylabel('s')
        plt.show()
        

def plotBezierCurves(controlPoints,degree):
    """
    Paramters:
        controlPoints : A 2D list of control points (example : [[x1,y1],[x2,y2], ... , [xn,yn]]
        degree : degree of bezier basis functions, degree = 1 for linear and so on
    Output:
        Plots the bezier curve controlled by a control polygon and also draws the control points
    """
    if (len(controlPoints) != degree +1):
        raise ValueError('The number of control points should be 1 more than the degree')
    controlPoints = np.array(controlPoints)
    cpX , cpY = controlPoints[:,0] , controlPoints[:,1]
    t = np.linspace(0,1,1000)
    basisFunctions = getBezierCurveBasisFunctions(degree)
    curve, i = 0,0
    for func in basisFunctions:
        curve += np.outer(eval(func, {'t':t}),controlPoints[i])
        i += 1
    curveX , curveY = curve[:,0] , curve[:,1]
    plt.plot(curveX,curveY, label = "Bezier Curve")
    plt.plot(cpX,cpY,'-og' , label = "Control Points")
    for i in range(len(cpX)):
        plt.annotate("P"+str(i), (cpX[i],cpY[i]), textcoords="offset points", xytext=(0,10), ha='center') 
    plt.title("Bezier curves for degree " + str(degree))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.xlim([min(cpX) - 0.3, max(cpX) + 0.3])
    plt.ylim([min(cpY) - 0.3, max(cpY) + 0.3])
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()

def cubicBezierCurvesJoinedWithContinuity(continuity = 'C-1', cpP=[[0,0],[0.5,1],[0.5,0],[1,1]], cpQ=[[1.5,1.5],[2,2.5],[2,1.5],[2.5,2.5]]):
    """
    Paramters:
        controlPoints cpP: A 2D list of control points for curve P (example : [[x1,y1],[x2,y2], ... , [xn,yn]]
        controlPoints cpQ: A 2D list of control points for curve Q (example : [[x1,y1],[x2,y2], ... , [xn,yn]]
        continuity : a string to specify the desired continuity. 
    Output:
        Plots the bezier curves joined by desired continuity.
    """
    cpP = np.array(cpP)
    cpQ = np.array(cpQ)
    cpPX , cpPY = cpP[:,0] , cpP[:,1]
    cpQX , cpQY = cpQ[:,0] , cpQ[:,1]
    t = np.linspace(0,1,1000)
    basisFunctions = getBezierCurveBasisFunctions(3)
    curveP, curveQ, i = 0,0,0
    if (continuity == 'C0'):
        cpQ[0] = cpP[3]
    elif (continuity == 'C1'):
        cpQ[0] = cpP[3]
        cpQ[1] = 2*cpP[3] - cpP[2]
    elif (continuity == 'C2'):
        cpQ[0] = cpP[3]
        cpQ[1] = 2*cpP[3] - cpP[2]
        cpQ[2] = cpP[1] - 4*cpP[2] + 4*cpP[3]
    for func in basisFunctions:
        curveP += np.outer(eval(func, {'t':t}),cpP[i])
        curveQ += np.outer(eval(func, {'t':t}),cpQ[i])
        i += 1
    curvePX , curvePY = curveP[:,0] , curveP[:,1]
    curveQX , curveQY = curveQ[:,0] , curveQ[:,1]
    plt.plot(curvePX,curvePY, label = "Bezier Curve P(t)")
    plt.plot(curveQX,curveQY, label = "Bezier Curve Q(t)")
    plt.plot(cpPX,cpPY,'--og' , label = "Control Points for P(t)")
    plt.plot(cpQX,cpQY,'--or' , label = "Control Points Q(t)")
    for i in range(len(cpPX)):
        plt.annotate("P"+str(i), (cpPX[i],cpPY[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate("Q"+str(i), (cpQX[i],cpQY[i]), textcoords="offset points", xytext=(0,-10), ha='center')
    plt.title("Cubic Bezier curves with " + continuity + " continuity")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()
    
def newBezierBasisToPassThroughCP(controlPoints,degree):
    """
    Paramters:
        controlPoints : A 2D list of control points (example : [[x1,y1],[x2,y2], ... , [xn,yn]]
        degree : degree of bezier basis functions. This function only accepts degrees which satisfy the inequality 2 <= degree <= 3
    Output:
        Plots the bezier curve controlled by a control polygon such that the curve passes through these points, and also draws the control points
    """
    if (len(controlPoints) != degree +1):
        raise ValueError('The number of control points should be 1 more than the degree')
    if (degree > 3 or degree < 2 ):
        raise ValueError('This function only works for quadratic and cubic bezier curves.')
    controlPoints = np.array(controlPoints)
    cpX , cpY = controlPoints[:,0] , controlPoints[:,1]
    t = np.linspace(0,1,1000)
    if (degree == 2):
        basisFunctions = ['2*(1-t)*(0.5-t)','4*(1-t)*t','-2*t*(0.5-t)']
    elif (degree == 3):
        basisFunctions = ['(9/2)*(1-t)*((1/3)-t)*((2/3)-t)','(27/2)*t*(1-t)*((2/3)-t)','(-27/2)*t*(1-t)*((1/3)-t)','(9/2)*t*((1/3)-t)*((2/3)-t)']
    curve, i = 0,0
    for func in basisFunctions:
        curve += np.outer(eval(func, {'t':t}),controlPoints[i])
        i += 1
    curveX , curveY = curve[:,0] , curve[:,1]
    plt.plot(curveX,curveY, label = "Bezier Curve")
    plt.plot(cpX,cpY,'-og' , label = "Control Points")
    for i in range(len(cpX)):
        plt.annotate("P"+str(i), (cpX[i],cpY[i]), textcoords="offset points", xytext=(0,10), ha='center') 
    plt.title("Bezier curves for degree " + str(degree))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.xlim([min(cpX) - 0.3, max(cpX) + 0.3])
    plt.ylim([min(cpY) - 0.3, max(cpY) + 0.3])
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()

def plotBezierSurfaces(controlPoints,degree):
    """
    Paramters:
        controlPoints : A 3D list of control points (example : [[x1,y1,z1],[x2,y2,z2], ... , [xn,yn,zn]]
        degree : degree of bezier basis functions, degree = 1 for linear and so on
    Output:
        Plots the bezier patch controlled by a control polygon and also draws the control points
    """
    if (len(controlPoints) != (degree +1)**2):
        raise ValueError('The number of control points should be (degree + 1)**2')
    controlPoints = np.array(controlPoints)
    cpX , cpY, cpZ = controlPoints[:,0] , controlPoints[:,1] , controlPoints[:,2]
    t = np.linspace(0,1,20)
    s = np.linspace(0,1,20)
    T, S = np.meshgrid(t, s)
    basisFunctions = getBezierSurfaceBasisFunctions(degree)
    fig = plt.figure()
    ax = ax = fig.gca(projection='3d')#Axes3D(fig)
    curve, i = 0,0
    for func in basisFunctions:
        curve += np.outer(eval(func,{'t':T,'s':S}),controlPoints[i])
        i += 1
    curveX , curveY , curveZ = curve[:,0] , curve[:,1] , curve[:,2]
    c = ax.plot_trisurf(curveX,curveY, curveZ, label = "Bezier Surface")
    c._facecolors2d=c._facecolors3d
    c._edgecolors2d=c._edgecolors3d
    ax.plot(cpX,cpY,cpZ,'--og' , label = "Control Points")
    for i in range(len(cpX)):
        ax.text(cpX[i],cpY[i],cpZ[i],"P"+str(i),horizontalalignment='left',verticalalignment='top') 
    plt.title("Bezier surface for degree " + str(degree))
    ax.legend()
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

def newBezierSurfacesThroughCP(controlPoints,degree):
    """
    Paramters:
        controlPoints : A 3D list of control points (example : [[x1,y1,z1],[x2,y2,z2], ... , [xn,yn,zn]]
        degree : degree of bezier basis functions. This function only accepts degrees which satisfy the inequality 2 <= degree <= 3
    Output:
        Plots the bezier patch controlled by a control polygon such that the curve passes through these points, and also draws the control points
    """
    if (len(controlPoints) != (degree +1)**2):
        raise ValueError('The number of control points should be (degree + 1)**2')
    if (degree > 3 or degree < 2 ):
        raise ValueError('This function only works for quadratic and cubic bezier curves.')
    controlPoints = np.array(controlPoints)
    cpX , cpY, cpZ = controlPoints[:,0] , controlPoints[:,1] , controlPoints[:,2]
    t = np.linspace(0,1,20)
    s = np.linspace(0,1,20)
    T, S = np.meshgrid(t, s)
    if (degree == 2):
        basisFunctions = ['2*(1-s)*(0.5-s)*2*(1-t)*(0.5-t)','2*(1-s)*(0.5-s)*4*(1-t)*t','2*(1-s)*(0.5-s)*(-2)*t*(0.5-t)',
                          '4*(1-s)*s*2*(1-t)*(0.5-t)','4*(1-s)*s*4*(1-t)*t','4*(1-s)*s*(-2)*t*(0.5-t)',
                          '(-2)*s*(0.5-s)*2*(1-t)*(0.5-t)','(-2)*s*(0.5-s)*4*(1-t)*t','(-2)*s*(0.5-s)*(-2)*t*(0.5-t)'
                          ]
    elif (degree == 3):
        basisFunctions = ['(9/2)*(1-s)*((1/3)-s)*((2/3)-s)*(9/2)*(1-t)*((1/3)-t)*((2/3)-t)','(9/2)*(1-s)*((1/3)-s)*((2/3)-s)*(27/2)*t*(1-t)*((2/3)-t)',
                          '(9/2)*(1-s)*((1/3)-s)*((2/3)-s)*(-27/2)*t*(1-t)*((1/3)-t)','(9/2)*(1-s)*((1/3)-s)*((2/3)-s)*(9/2)*t*((1/3)-t)*((2/3)-t)',
                          
                          '(27/2)*s*(1-s)*((2/3)-s)*(9/2)*(1-t)*((1/3)-t)*((2/3)-t)','(27/2)*s*(1-s)*((2/3)-s)*(27/2)*t*(1-t)*((2/3)-t)',
                          '(27/2)*s*(1-s)*((2/3)-s)*(-27/2)*t*(1-t)*((1/3)-t)','(27/2)*s*(1-s)*((2/3)-s)*(9/2)*t*((1/3)-t)*((2/3)-t)',

                          '(-27/2)*s*(1-s)*((1/3)-s)*(9/2)*(1-t)*((1/3)-t)*((2/3)-t)','(-27/2)*s*(1-s)*((1/3)-s)*(27/2)*t*(1-t)*((2/3)-t)',
                          '(-27/2)*s*(1-s)*((1/3)-s)*(-27/2)*t*(1-t)*((1/3)-t)','(-27/2)*s*(1-s)*((1/3)-s)*(9/2)*t*((1/3)-t)*((2/3)-t)',

                          '(9/2)*s*((1/3)-s)*((2/3)-s)*(9/2)*(1-t)*((1/3)-t)*((2/3)-t)','(9/2)*s*((1/3)-s)*((2/3)-s)*(27/2)*t*(1-t)*((2/3)-t)',
                          '(9/2)*s*((1/3)-s)*((2/3)-s)*(-27/2)*t*(1-t)*((1/3)-t)','(9/2)*s*((1/3)-s)*((2/3)-s)*(9/2)*t*((1/3)-t)*((2/3)-t)'
                          ]
    fig = plt.figure()
    ax = Axes3D(fig)
    curve, i = 0,0
    for func in basisFunctions:
        curve += np.outer(eval(func,{'t':T,'s':S}),controlPoints[i])
        i += 1
    curveX , curveY , curveZ = curve[:,0] , curve[:,1] , curve[:,2]
    c = ax.plot_trisurf(curveX,curveY, curveZ, label = "Bezier Surface")
    c._facecolors2d=c._facecolors3d
    c._edgecolors2d=c._edgecolors3d
    ax.plot(cpX,cpY,cpZ,'--og' , label = "Control Points")
    for i in range(len(cpX)):
        ax.text(cpX[i],cpY[i],cpZ[i],"P"+str(i),horizontalalignment='left',verticalalignment='top') 
    plt.title("Bezier surface through control points for degree " + str(degree))
    ax.legend()
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

def quadraticBezierPatchesJoinedWithContinuity(continuity = 'C-1',cpP=[[0,0,10],[3,0,-10],[6,0,0],[0,3,12],[3,3,0],[6,3,0],[0,6,0],[3,6,4],[6,6,0]],cpQ=[[10,10,10], [13,10,-10], [16,10,0], [10,13,12], [13,13,0], [16,13,0], [10,16,0], [13,16,4],[16,16,0]]):
    """
    Paramters:
        controlPoints cpP: A 3D list of control points for patch P (example : [[x1,y1,z1],[x2,y2,z2], ... , [xn,yn,zn]]
        controlPoints cpQ: A 3D list of control points for patch Q (example : [[x1,y1,z1],[x2,y2,z2], ... , [xn,yn,zn]]
        continuity : a string to specify the desired continuity. 
    Output:
        Plots the bezier patches joined by desired continuity.
    """
    cpP = np.array(cpP)
    cpQ = np.array(cpQ)
    cpPX , cpPY , cpPZ = cpP[:,0] , cpP[:,1] , cpP[:,2]
    cpQX , cpQY , cpQZ = cpQ[:,0] , cpQ[:,1] , cpQ[:,2]
    t = np.linspace(0,1,10)
    s = np.linspace(0,1,10)
    T, S = np.meshgrid(t, s)
    basisFunctions = getBezierSurfaceBasisFunctions(2)
    fig = plt.figure()
    ax = ax = fig.gca(projection='3d')
    curveP, curveQ, i = 0,0,0
    if (continuity == 'C0'):
        cpQ[0] = cpP[-3]
        cpQ[1] = cpP[-2]
        cpQ[2] = cpP[-1]
    elif (continuity == 'C1'):
        cpQ[0] = cpP[-3]
        cpQ[1] = cpP[-2]
        cpQ[2] = cpP[-1]
        cpQ[3] = cpP[-1] + cpP[-3] - cpP[-4]
    for func in basisFunctions:
        curveP += np.outer(eval(func,{'t':T,'s':S}),cpP[i])
        curveQ += np.outer(eval(func,{'t':T,'s':S}),cpQ[i])
        i += 1
    curvePX , curvePY , curvePZ = curveP[:,0] , curveP[:,1] , curveP[:,2]
    curveQX , curveQY , curveQZ = curveQ[:,0] , curveQ[:,1] , curveQ[:,2]

    c = ax.plot_trisurf(curvePX,curvePY, curvePZ, label = "Bezier surface P(s,t)")
    c._facecolors2d=c._facecolors3d
    c._edgecolors2d=c._edgecolors3d

    c = ax.plot_trisurf(curveQX,curveQY, curveQZ, label = "Bezier surface Q(s,t)")
    c._facecolors2d=c._facecolors3d
    c._edgecolors2d=c._edgecolors3d
    plt.title("Quadratic Bezier curves with " + continuity + " continuity")
    ax.legend()
    plt.xlabel('t')
    plt.ylabel('s')
    plt.show()

def parseFile(filename):
    """
    Parameters:
        filename: .txt file to read
    Output:
        vertices: numpy array containing vertices for example [[0.93,1.993],[8.34,4.44]]
        faces: dictionary of faces containing list of vertices as values, for example {0:[0,1,2],1:[3,4,5]}
    """
    file = open(filename,"r")
    data = file.readlines()
    nv,nf = [int(s) for s in data[0].split() if s.isdigit()]
    vertices = np.array([list(map(float, vertex.split())) for n,vertex in enumerate(data[1:nv+1])])
    faces = {n:list(map(int, face.split())) for n,face in enumerate(data[1+nv:])}
    return vertices,faces

#####Refer to other two code files ####


def circle(filename):
    """
    Parameters:
        filename: .txt file to read
    Output:
        Plots the approximated circle.
    """
    vertices,faces = parseFile(filename)
    vX , vY = vertices[:,0] , vertices[:,1]
    t = np.linspace(0,1,1000)
    basisFunctions = getBezierCurveBasisFunctions(2)
    for nFace,lFace in faces.items():
        curve,i = 0, 0
        for func in basisFunctions:
            curve += np.outer(eval(func, {'t':t}),vertices[lFace[i]])
            i += 1
        curveX , curveY = curve[:,0] , curve[:,1]
        plt.plot(curveX,curveY,color='blue')
    
    plt.plot(vX,vY,'--og' , label = "Control Points")
    plt.title("Bezier curves for 2")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=.4,linestyle='--')
    plt.legend()
    plt.show()

def sphere(filename):
    """
    Parameters:
        filename: .txt file to read
    Output:
        Plots the approximated sphere.
    """
    vertices,faces = parseFile(filename)
    vX , vY, vZ = vertices[:,0] , vertices[:,1] , vertices[:,2]
    t = np.linspace(0,1,10)
    s = np.linspace(0,1,10)
    T, S = np.meshgrid(t, s)
    basisFunctions = getBezierSurfaceBasisFunctions(2)
    fig = plt.figure()
    ax = Axes3D(fig)
    x , y , z = 0,0,0
    for nFace,lFace in faces.items():
        curve,i = 0, 0
        for func in basisFunctions:
            curve += np.outer(eval(func,{'t':T,'s':S}),vertices[lFace[i]])
            i += 1
        curveX , curveY , curveZ = curve[:,0] , curve[:,1] , curve[:,2]
        x = np.append(x,curveX)
        y = np.append(y,curveY)
        z = np.append(z,curveZ)
    c = ax.plot_trisurf(x,y, z)
    
    ax.plot(vX,vY,vZ,'og' , label = "Control Points")
    plt.title("Bezier curves for 2")
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.grid(alpha=.4,linestyle='--')
    plt.show()


knot_vector = [0,1,2,3,4,5,6]
#plotBsplineBasisFunctions(knot_vector,5)         
#plotBezierBasisFunctions(type='curve',degree=2)
#plotBezierBasisFunctions(type='surface',degree=1)
curveCP = {1:[[0,0],[0.5,1]], 2: [[1,2],[0,0],[2,0]],3:[[0,0],[0.5,1],[0.5,0],[1,1]],4:[[0,0],[1,0],[1,1],[0,1],[0.5,2]]}   
#plotBezierCurves(curveCP[2],2)    
#newBezierBasisToPassThroughCP(curveCP[3],3)
#w = [[0,0,0],[3,0,0],[6,0,0],[0,3,0],[3,3,0],[6,3,0],[0,6,0],[3,6,0],[6,6,0]]
#c = [[0,1.5,1.5],[0,1.84,0.76],[0,1.38,0.57],[0,1,1],[0,1.5,1],[0,1.6,0.6],[0,1.2,0.81],[0,1.2,1.2],[0,1.44,0.96]]
#d = [[0.38393,0,10.3999],[3,0,-10.8383],[6.393994,0,0],[0,3,12],[3,3.39993,0],[6,3.3993,0],[0,6,0],[3,6,4],[6,6,0]]
patchCP = {2:[[0,0,10],[3,0,-10],[6,0,0],[0,3,12],[3,3,0],[6,3,0],[0,6,0],[3,6,4],[6,6,0]],
           1 : [[0,0,0],[3,0,0],[0,3,0],[3,3,0]],
           3: [[0,0,0], [1,0,2], [2,0,-1],[3,0,0],
                 [0,1,0], [1,1,4], [2,1,4], [3,1,0],
                 [0,2,0], [1,2,8], [2,2,8],  [3,2,0],
                 [0,3,0], [1,3,0], [2,3,0], [3,3,0]],
           4: [[0,0,0], [1,0,2], [2,0,-1],[3,0,0],
                 [0,1,0], [1,1,4], [2,1,4], [3,1,4], [3,2,4], [3,1,0],
                 [0,2,0], [1,2,8], [2,2,8], [2,4,8], [4,2,8],  [3,2,0],
                 [0,3,0], [1,3,0], [2,3,0], [3,3,0], [3,2,1],
                 [0,3,1], [1,3,1], [2,3,1], [3,3,1]
           ]}
#plotBezierSurfaces(w,2)
#newBezierSurfacesThroughCP(p,2)
#cubicBezierCurvesJoinedWithContinuity(continuity = 'C2',cpP = [[0,0],[0.5,1],[0.5,0],[1,1]],cpQ = [[1.5,1.5],[2,2.5],[2,1.5],[2.5,2.5]])
#circle("../resources/circle.txt")
#sphere("../resources/sphere.txt")
#plotBezierSurfaces(patchCP[4],4)
#newBezierSurfacesThroughCP(patchCP[3],3)
cpP = [[0,0,10],[3,0,-10],[6,0,0],[0,3,12],[3,3,0],[6,3,0],[0,6,0],[3,6,4],[6,6,0]]
cpQ = [[10,10,10], [13,10,-10], [16,10,0], [10,13,12], [13,13,0], [16,13,0], [10,16,0], [13,16,4],[16,16,0]]
#quadraticBezierPatchesJoinedWithContinuity(continuity='C1',cpP=cpP,cpQ=cpQ)
