
'''
	Rosenbrock Function: https://www.sfu.ca/~ssurjano/rosen.html

	Objective: Min

	Bounds:
		Upper:  2048.00
		LoweR: -2048.00

	Dimension: 30
'''

from numba import cuda

@cuda.jit(device= True)
def get_fitness(x):

	top = 0.0

	aux = len(x) - 1

	for i in range(0, aux):
		xi = x[i]
		xnext = x[i + 1]
		
		new = 100 * (xnext - xi ** 2) ** 2 + (xi - 1) ** 2
		top = top + new

	return top

def get_dim():
	return 30

def get_bounds():
	return 2048.00, -2048.00

def is_constrained():
	return False