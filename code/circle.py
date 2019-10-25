import matplotlib.pyplot as plt 
import math
class ControlPoint(object):
    def __init__(self, x_val, y_val):
        self.x = float(x_val)
        self.y = float(y_val)

    def coordinates(self):
        return [self.x, self.y]
    
    def __mul__(self, integer):
        return ControlPoint (self.x * integer, self.y * integer)

    def __add__(self, point):
        return ControlPoint (self.x + point.x, self.y + point.y)

class Face(object):
    def __init__(self, controlPoints):
        self._P = controlPoints #List of Control Points 

    #Quadratic Basis functions 
    def _b0(self, t):
        return (1 - t)**2
    def _b1(self, t):
        return 2*(1 - t)*t
    def _b2(self, t):
        return t**2

    def get_all_x(self):
        #returns x coordinates of each control point of the face, in a list 
        return [self._P[i].x for i in range(3)]
        
    def get_all_y(self):
        #returns y coordinates of each control point of the face, in a list 
        return [self._P[i].y for i in range(3)]
        
    def P(self, t):
        return (self._P[0] * self._b0(t)) + (self._P[1] * self._b1(t)) + (self._P[2] * self._b2(t))

def frange(m, n, o):
    #Custom floating point Range function. m, n inclusive 
    a = [m]
    while a[-1] <= n:
        if a[-1] + o > n:
            return a
        a.append(a[-1] + o)
    return a

def circle():
    mesh = open("../resources/circle.txt")
    #first line tells how many vertices and how many faces(lines)
    count = [int(i.strip()) for i in mesh.readline().strip().split() ]

    points = {}
    #loading points 
    for j in range(count[0]):
        p = [i for i in mesh.readline().strip().split()]
        points["P"+str(j)] = ControlPoint(p[0], p[1])

    AllFaces = {}
    #loading faces
    for j in range(count[1]):
        #list containing ith face's control points
        f = [points["P"+i] for i in mesh.readline().strip().split()] 
        AllFaces["F"+str(j)] = Face(f)
    
    #Line Mesh
    x_t = []
    y_t = []

    #Control Points 
    x_a = []
    y_a = []

    step = 0.0001 #difference in each t 
    
    for face in AllFaces:
        
        x_a.extend(AllFaces[face].get_all_x())
        y_a.extend(AllFaces[face].get_all_y())
        
        for t in frange(0, 1, step):
            cord = AllFaces[face].P(t)
            x_t.append(cord.x)
            y_t.append(cord.y)

    GolRatio = (1 + math.sqrt(5))/2
    plt.plot(x_a, y_a, 'x--', label='control points') #Dashed Control Points 
    plt.plot(x_t, y_t, 'r', linewidth=GolRatio)
    plt.grid(linestyle='-')
    legend = plt.legend()
    plt.show()  

circle()
