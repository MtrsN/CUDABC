import numpy as np
import fitness as fit
from numba import cuda

from parameters import *

# ABC Function
@cuda.jit(device=True)
def ABC_classic_function(bee, neigh, random_weight):
	return bee + random_weight * (bee - neigh)

# Fitness Initialization
@cuda.jit(debug=True)
def ABC_initialization(hive, fitness, global_fitness):

	i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

	if(i < bees):
		fitness[i] = fit.fitness(hive[i])

	cuda.syncthreads()

	cuda.atomic.min(global_fitness, 0, fitness[i])

# Classic CUDABC
@cuda.jit(debug=True)
def ABC_optimization(hive, fitness,
					 limit, limit_index, 
					 neigh_index, neigh_onlooker_index, 
					 food_index, food_onlooker_index, 
					 weights_index, weights_onlooker_index, 
					 tournament_index, global_fitness):

	i = cuda.threadIdx.x

	for generation in range(0, generations):

		# Employed Cycle
		if(i < bees):
			oldIndex = hive[i, food_index[generation, i]]

			hive[i, food_index[generation, i]] = ABC_classic_function(hive[i, food_index[generation, i]], # i-th / food
																		hive[neigh_index[generation, i], food_index[generation, i]], # neigh / food
																		weights_index[generation, i]) # random weight

			newFitness = fit.fitness(hive[i])

			if(newFitness < fitness[i]):
				fitness[i] = newFitness
			else:
				hive[i, food_index[generation, i]] = oldIndex
				limit[i] = limit[i] + 1

		# Onlooker Cycle
		if(i < int(bees/2)):

			selected_tournament = -1

			if(fitness[tournament_index[generation, i, 0]] < fitness[tournament_index[generation, i, 1]]):
				selected_tournament = tournament_index[generation, i, 0]
			else:
				selected_tournament = tournament_index[generation, i, 1]

			oldIndex = hive[selected_tournament, food_onlooker_index[generation, i]]

			hive[selected_tournament, food_onlooker_index[generation, i]] = ABC_classic_function(hive[selected_tournament, food_onlooker_index[generation, i]], # tournament / food
																									hive[neigh_onlooker_index[generation, i], food_onlooker_index[generation, i]], # neigh / food
																									weights_onlooker_index[generation, i]) # random weight

			newFitness = fit.fitness(hive[selected_tournament])

			if(newFitness < fitness[selected_tournament]):
				fitness[selected_tournament] = newFitness
			else:
				hive[selected_tournament, food_onlooker_index[generation, i]] = oldIndex
				limit[i] = limit[i] + 1

		cuda.syncthreads()

		# Scout Cycle
		if(i < bees):

			if(limit[i] >= limit_solutions):

				oldIndex = hive[i, food_index[generation, i]]

				hive[i, food_index[generation, i]] = limit_index[generation, i]
				
				newFitness = fit.fitness(hive[i])

				if(newFitness < fitness[i]):
					fitness[i] = newFitness
					limit[i] = 0
				else:
					hive[i, food_index[generation, i]] = oldIndex
					limit[i] = limit[i] + 1

		cuda.syncthreads()
		cuda.atomic.min(global_fitness, generation, fitness[i])

