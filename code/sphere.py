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
	def _b00(self, s, t):
		return (((1 - s)**2)*(1 - t)**2)
	def _b01(self, s, t):
		return (2 * (1 - t) * t * ((1 - s)**2) )
	def _b02(self, s, t):
		return (( (1 - s)**2) * ((t)**2) )
	def _b10(self, s, t):
		return (2 * (1 - s) * s * ((1 - t)**2) )
	def _b11(self, s, t):
		return (4 * (1 - s) * (1 - t) * s * t)
	def _b12(self, s, t):
		return (2 * (1 - s) * s * (t**2))
	def _b20(self, s, t):
		return (( (1 - t)**2) * ((s)**2) )
	def _b21(self, s, t):
		return (2 * (1 - t) * t * (s**2))
	def _b22(self, s, t):
		return ((s**2)*(t**2))

	def P(self, s, t):
		ij = [TDPoint() for i in range(9)]
		ij[0] = self._P[0] * self._b00(s, t)
		ij[4] = self._P[4] * self._b01(s, t)
		ij[1] = self._P[1] * self._b02(s, t)
		ij[7] = self._P[7] * self._b10(s, t)
		ij[8] = self._P[8] * self._b11(s, t)
		ij[5] = self._P[5] * self._b12(s, t)
		ij[3] = self._P[3] * self._b20(s, t)
		ij[6] = self._P[6] * self._b21(s, t)
		ij[2] = self._P[2] * self._b22(s, t)

		p_st = TDPoint()

		for i in range(9):
			p_st += ij[i] 
		
		return p_st


	def get_all_x(self):
		#returns x coordinates of each control point of the face, in a list 
		return [self._P[i].x for i in range(9)]
		
	def get_all_y(self):
		#returns y coordinates of each control point of the face, in a list 
		return [self._P[i].y for i in range(9)]

	def get_all_z(self):
		#returns z coordinates of each control point of the face, in a list 
		return [self._P[i].z for i in range(9)]

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
		AllFaces["F"+str(j)] = Face(f)

	# for i in AllFaces:
	# 	print(i)
	# 	for j in AllFaces[i]._P:
	# 		print(j.coordinates())

	#Surface Mesh
	x_st = []
	y_st = []
	z_st = [] 

	#Control Points 
	x_a = []
	y_a = []
	z_a = []

	#difference in each t and s
	ds = np.arange(0, 1, 0.1)
	dt = np.arange(0, 1, 0.1)

	for face in AllFaces:
		x_a.extend(AllFaces[face].get_all_x())
		y_a.extend(AllFaces[face].get_all_y())
		z_a.extend(AllFaces[face].get_all_z())
		for s in ds:
			for t in dt:
				cord = AllFaces[face].P(s, t)
				x_st.append(cord.x)
				y_st.append(cord.y)
				z_st.append(cord.z)

	# GolRatio = (1 + math.sqrt(5))/2
	tdPlot = plt.figure()
	axs = tdPlot.add_subplot(111, projection='3d')

	axs.plot_wireframe(np.array([x_a]), np.array([y_a]), np.array([z_a]))
	plt.show()


sphere()

