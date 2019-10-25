from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt 
import numpy as np

class TDPoint(object):
	"""Three Dimensional Point Object"""
	def __init__(self, x_val=0, y_val=0, z_val=0):
		self.x = float(x_val)
		self.y = float(y_val)
		self.z = float(z_val)

	def coordinates(self):
		return [self.x, self.y, self.z]
	
	def __mul__(self, scalar):
		return TDPoint (self.x * scalar, self.y * scalar, self.z * scalar)

	def __add__(self, point):
		return TDPoint (self.x + point.x, self.y + point.y, self.z * point.z)


class Face(object):
	def __init__(self, points):
		self._P = points #List of Control Points 

	#Quadratic bezier patch basis functions
	def b(self, ij, st):
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
		p_st = TDPoint()
		for i in range(len(self._P)):
			p_st +=  (self._P[j[i]] * self.b(i // 3, s) * self.b(i % 3, t))

		return p_st	


def sphere():
	mesh = open("../resources/sphere.txt")
	count = [ int(i.strip()) for i in mesh.readline().strip().split()]

	points = {}
	#loading points 
	for j in range(count[0]):
		p = [i for i in mesh.readline().strip().split()]
		points["P"+str(j)] = TDPoint(p[0], p[1], p[2])

	# for i in points:
	# 	print(i, points[i].coordinates())

	AllFaces = {}
	#loading faces
	for j in range(count[1]):
		#list containing ith face's control points
		f = [points["P"+i] for i in mesh.readline().strip().split()]
		f = [f[0], f[4], f[1], f[5], f[2], f[6], f[3], f[7], f[0]]

		AllFaces["F"+str(j)] = Face(f)

	# for i in AllFaces:
	# 	print(i)
	# 	for j in AllFaces[i]._P:
	# 		print(j.coordinates())

	#Surface Mesh
	x_st, y_st, z_st = [], [], []

	#Control Points 
	x_a, y_a, z_a = [], [], []

	for face in AllFaces:
		x_a.extend(AllFaces[face].get_all_x())
		y_a.extend(AllFaces[face].get_all_y())
		z_a.extend(AllFaces[face].get_all_z())
		
		for s in np.arange(0, 1.1, 0.1):
			for t in np.arange(0, 1.1, 0.1):
					cord = AllFaces[face].P(s, t)
					x_st.append(cord.x)
					y_st.append(cord.y)
					z_st.append(cord.z)



	# GolRatio = (1 + math.sqrt(5))/2
	tdPlot = plt.figure()
	axs = tdPlot.add_subplot(111, projection='3d')
	axs.plot_wireframe(np.array([x_a]), np.array([y_a]), np.array([z_a])) #original wireframs
	# axs.plot_wireframe(np.array([x_st]), np.array([y_st]), np.array([z_st])) #generate wireframe
	plt.show()


sphere()

