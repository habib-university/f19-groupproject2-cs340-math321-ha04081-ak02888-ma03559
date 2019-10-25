import matplotlib.pyplot as plt 

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
		ij = [0 for i in range(9)]
		ij[0] = self._b00(s, t) * self._P[0]
		ij[1] = self._b01(s, t) * self._P[1]
		ij[2] = self._b02(s, t) * self._P[2]
		ij[3] = self._b10(s, t) * self._P[3]
		ij[4] = self._b11(s, t) * self._P[4]
		ij[5] = self._b12(s, t) * self._P[5]
		ij[6] = self._b20(s, t) * self._P[6]
		ij[7] = self._b21(s, t) * self._P[7]
		ij[8] = self._b22(s, t) * self._P[8]
		return sum(ij)

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

	


sphere()