import matplotlib.pyplot as plt

import os
import parameters as P

from numpy import *
from decimal import Decimal

def generate_plot(globals):

	plt.plot(globals)
	plt.ylabel("Fitness")
	plt.xlabel("Generations")
	plt.show()

def get_mean(x):

	sum = 0.0

	for i in range(0, len(x)):
		sum += x[i]

	return sum / len(x)

def get_statics(x):

	md = get_mean(x)

	var = 0.0

	for i in range(0, len(x)):

		sub = x[i] - md
		quad = sub * sub
		var += quad

	return md, sqrt(var / (len(x) - 1)), amin(x), amax(x) # mean, sd, best, worst

def get_convergence(x):

	y = empty(x.shape[1])

	for i in range(0, x.shape[1]):

		y[i] = Decimal(average(x[:, i]))

	return y

def write_statistical(mean, sd, worst, best, name):

	dirDataset = "Fitness/" + P.function.capitalize()

	if(not os.path.exists(dirDataset)):
		os.makedirs(dirDataset)

	deepDir= "/StatisticalData_" + name + ".txt" 

	finalDir = dirDataset + deepDir

	file = open(finalDir, "w")

	file.write("Mean %s: %s\n" % (name, mean))
	file.write("Standart Derivation %s: %s\n" % (name, sd))
	file.write("Best %s Found: %s\n" % (name, best))
	file.write("Worst %s Found: %s\n" % (name, worst))

	file.close()

def write_convergence(y):

	name = "Fitness/" + P.function.capitalize() + "/Convergence.txt"

	file = open(name, "w")

	for i in y:
		file.write(str(Decimal(i)).replace(".", ",") + "\n")

	file.close()