
import numpy as np
import fitness as fit
import cuda as cu
import timeit
import statistic as stt

if __name__ == '__main__':
	
	# Fitness
	dim = fit.get_dim()
	upper_bound, lower_bound = fit.get_bounds()

	# Statistical
	curve_statistic = np.empty((cu.runs, cu.generations))
	data_statistic = np.empty(cu.runs)
	time_statistic = np.empty(cu.runs)

	# runs
	for run in range(0, cu.runs):
        
		print("Starting run %d" % (run + 1), end= "\n\n")

		np.random.seed( (run + run) * (run + run) )

		start = timeit.default_timer() # Timer 1

		# Population
		hive = np.random.uniform(lower_bound, upper_bound, (cu.bees, dim))
		fitness = np.zeros(cu.bees)
		limit = np.zeros(cu.bees)

		# RNG Index
		neigh_index = np.random.randint(0, cu.bees, size=(cu.generations, cu.bees))
		neigh_onlooker_index = np.random.randint(0, cu.bees, size=(cu.generations, int(cu.bees / 2)))

		food_index = np.random.randint(0, dim, size=(cu.generations, cu.bees))
		food_onlooker_index = np.random.randint(0, dim, size=(cu.generations, int(cu.bees / 2)))

		weights_index = np.random.uniform(size=(cu.generations, cu.bees))
		weights_onlooker_index = np.random.uniform(size=(cu.generations, int(cu.bees/2)))

		tournament_index = np.random.uniform(0, cu.bees, size=(cu.generations, int(cu.bees / 2), 2)).astype(np.int32)
		limit_index = np.random.uniform(lower_bound, upper_bound, size=(cu.generations, cu.bees))

		# Globals
		global_fitness = np.array([9999999999999999] * cu.generations).astype(np.float64)
		global_solution = np.zeros(dim)

		# Kernels
		cu.ABC_initialization[1, cu.bees](hive, fitness, global_fitness)

		cu.ABC_optimization[1, cu.bees](hive, fitness, 
										limit, limit_index,
										neigh_index, neigh_onlooker_index, 
										food_index, food_onlooker_index, 
										weights_index, weights_onlooker_index, 
										tournament_index, global_fitness)

		stop = timeit.default_timer() # Timer 2

		curve_statistic[run] = global_fitness
		data_statistic[run] = global_fitness[-1]

		time_statistic[run] = stop - start

	mean_e, sd_e, best_e, worst_e = stt.get_statics(data_statistic) # Error
	mean_a, sd_a, best_a, worst_a = stt.get_statics(time_statistic) # Accuracy

	# Write
	stt.write_statistical(mean_e, sd_e, worst_e, best_e, "Fitness") # Write Fitness
	stt.write_statistical(mean_a, sd_a, worst_a, best_a, "Time") # Write Time
	stt.write_convergence(stt.get_convergence(curve_statistic))

	print("Mean Time: %.6f seconds" % (stt.get_mean(time_statistic)))
	print("Mean F(x): %.6f fitness" % (stt.get_mean(data_statistic)))

	stt.generate_plot(stt.get_convergence(curve_statistic))
