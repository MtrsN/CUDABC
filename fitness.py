from numba import cuda
import numpy as np

# ------------------------ Sphere Function ------------------------ #
@cuda.jit(device= True)
def fitness(x):

	sm = 0.0

	for i in range(0, len(x)):
		sm += x[i] * x[i]

	return sm

def get_dim():
	return 30

def get_bounds():
	return 100, -100
