
import parameters as P
import importlib

# ------------------------ Function Parameters ------------------------ #

function_dir 			= "problems." + P.function.lower()

function_fitness 		= "get_fitness"

function_bounds 		= "get_bounds"

function_dim 			= "get_dim"

function_constrained 	= "is_constrained"

function_restrictions 	= "get_restrictions"

# ----------------------------- Function ------------------------------ #

fitness 		= getattr(importlib.import_module(function_dir), function_fitness)

get_dim 		= getattr(importlib.import_module(function_dir), function_dim)

get_bounds		= getattr(importlib.import_module(function_dir), function_bounds)


