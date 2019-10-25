from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

class TDPoint(object):
	"""Three Dimensional Point Object"""
	def __init__(self, x_val, y_val, z_val):
		self.x = float(x_val)
		self.y = float(y_val)
		self.z = float(z_val)

	def coordinates(self):
		return [self.x, self.y, self.z]
	
	def __mul__(self, scalar):
		return TDPoint (self.x * scalar, self.y * scalar, self.z * scalar)

	def __add__(self, point):
		return TDPoint (self.x + point.x, self.y + point.y, self.z + point.z)

class Face(object):
	def __init__(self, points):
		self._P = points #List of Control Points 

	#Quadratic bezier patch basis functions
	def basis(self, ij, st):
		if ij == 0:

			return (1 - 2*st + st**2) 
		elif ij == 1:
			return (2*st - 2*st**2)
		elif ij == 2:
			return (st**2)
	
	def get_all_x(self):
		#returns x coordinates of each control point of the face, in a list 
		return [self._P[i].x for i in range(len(self._P))]
		
	def get_all_y(self):
		#returns y coordinates of each control point of the face, in a list 
		return [self._P[i].y for i in range(len(self._P))]

	def get_all_z(self):
		#returns z coordinates of each control point of the face, in a list 
		return [self._P[i].z for i in range(len(self._P))]
	
	def P(self, s, t):
		j = [0, 4, 1, 5, 2, 6, 3, 7, 8]
		p_st = TDPoint(0, 0, 0)
		for i in range(len(self._P)):
			p_st = (self._P[(i // 3) + (i % 3)] * self.basis(i // 3, s) * self.basis(i % 3, t)) + p_st

		return p_st	

def sphere():
	mesh = open("../resources/sphere.txt")
	count = [ int(i.strip()) for i in mesh.readline().strip().split()]

	points = {}
	#loading points 
	for j in range(count[0]):
		p = [i for i in mesh.readline().strip().split()]
		points["P"+str(j)] = TDPoint(p[0], p[1], p[2])

	AllFaces = {}
	#loading faces
	for j in range(count[1]):
		#list containing ith face's control points
		f = [points["P"+i] for i in mesh.readline().strip().split()]
		f = [f[0], f[4], f[1], f[5], f[2], f[6], f[3], f[7], f[0]] #rearranging according to figure 4 in question paper 
		AllFaces["F"+str(j)] = Face(f)

	mesh.close()

	#Surface Mesh
	x_st, y_st, z_st = [], [], []

	#Control Points 
	x_a, y_a, z_a = [], [], []

	step = 0.01
	for face in AllFaces:

		#Gather x, y, z coordinates of all vertices in an array for each 
		x_a.extend(AllFaces[face].get_all_x())
		y_a.extend(AllFaces[face].get_all_y())
		z_a.extend(AllFaces[face].get_all_z())
		
		for s in np.arange(0, 1 + step, step):
			for t in np.arange(0, 1 + step, step):
					cord = AllFaces[face].P(s, t) #Bezier Surface coordinate 
					x_st.append(cord.x)
					y_st.append(cord.y)
					z_st.append(cord.z)

	GolRatio = (1 + math.sqrt(5))/2

	tdPlot = plt.figure()
	axs = tdPlot.add_subplot(111, projection='3d')

	axs.plot_wireframe(np.array([x_a]), np.array([y_a]), np.array([z_a]), color='red', label='control points') #original wireframs
	axs.plot_wireframe(np.array(x_st), np.array(y_st), np.array([z_st]),  color='orange', linewidth=GolRatio, antialiased=False) #generate wireframe
	
	axs.set_title('Bezier Surface Plot')
	
	axs.set_xlabel('X-axis')
	axs.set_ylabel('Y-axis')
	axs.set_zlabel('Z-axis')

	legend = plt.legend()

	plt.show()

sphere()