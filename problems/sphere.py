
'''
	Sphere Function: https://www.sfu.ca/~ssurjano/spheref.html

	Objective: Min

	Bounds:
		Upper:  100.00
		LoweR: -100.00

	Dimension: 30
'''

from numba import cuda

@cuda.jit(device= True)
def get_fitness(x):

	sm = 0.0

	for i in range(0, len(x)):
		sm += x[i] * x[i]

	return sm

def get_dim():
	return 30

def get_bounds():
	return 100.0, -100.0

def is_constrained():
	return False

'''
@cuda.jit(device= True)
def fitness(x):

	top = 0.0

	aux = len(x) - 1

	for i in range(0, aux):
		xi = x[i]
		xnext = x[i + 1]
		
		new = 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2
		top = top + new

	return top
'''
