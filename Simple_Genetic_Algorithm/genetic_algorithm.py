"""
Genetic Algorithm Framework

This module provides a framework for solution finding using a Genetic Algorithm (GA).
Like hyperparameter optimization in ML.

Author: Tobia Ippolito
"""

# imports
import random
import copy
from datetime import datetime as dt
from joblib import Parallel, delayed
import enum
from abc import ABC, abstractmethod

def get_random(*args):
    """
    A helper function to create the get_random_param_value method.

    Parameters:
    - args: Variable number of arguments to determine the range or list for random value generation.

    Returns:
    - float or int: A random value based on the provided arguments. Also a element of a list is possible.
    """
    if len(args) == 0:
        return random.random()
    else:
        if type(args[0]) == int:
            if len(args) == 1:
                return random.randint(0, args[0])
            else:
                return random.randint(args[0], args[1])
        elif type(args[0]) == float:
            if len(args) == 1:
                return random.uniform(0.0, args[0])
            else:
                return random.uniform(args[0], args[1])
        elif type(args[0]) == list:
            return random.choice(args[0])
        else:
            return random.random()

class GA(ABC):
    """
    Abstract base class for Genetic Algorithm.

    Attributes:
    - generations (int): Number of generations.
    - population_size (int): Size of the population.
    - mutation_rate (float): Probability of mutation.
    - list_of_params (list): List of hyperparameters.
    - initial_solutions (list): Initial solutions for the population.
    - parallel (bool): Flag to enable parallel fitness calculation.
    - should_print (bool): Flag to control printing during optimization.

    Methods:
    - add_initial_solution(solution: dict) -> None:
        Adds an initial solution to the population.

    - get_short_duration_representation(start, end) -> str:
        Calculates and formats the duration between two time points.

    - calculate_fitness(kwargs, params) -> float:
        Abstract method to calculate fitness for an individual.

    - get_random_param_value(param_key) -> float or int:
        Abstract method to get a random value for a random individuals.

    - init_population(population_size) -> list:
        Initializes the population with random individuals.

    - select_individuals(population, population_size) -> list:
        Selects the best individuals from the population.

    - crossover(parents, possibility=0.5) -> list:
        Performs crossover to create offspring.

    - mutate_population(population, mutation_rate, population_size) -> list:
        Mutates the population based on the mutation rate.

    - add_newcomers(population, population_size) -> list:
        Adds new individuals to the population.

    - calculate_fitness_wrapper(kwargs, cur_individual, idx, len_population) -> tuple:
        Wrapper for parallel fitness calculation.

    - optimize(**kwargs) -> tuple:
        Optimizes hyperparameters using a Genetic Algorithm.
    """

    def __init__(self, generations, population_size, mutation_rate, list_of_params, 
                 initial_solutions=list(), parallel=True, should_print=True,
                 # Mutation
                 mutation_method='random', 
                 gaussian_std_dev=None, 
                 uniform_min=None, 
                 uniform_max=None, 
                 discrete_choices=None,
                 # Crossover
                 crossover_possiblity=0.5, 
                 crossover_method='uniform', 
                 crossover_points=(0, 1),
                 # Selection
                 selection_method='tournament', 
                 k=0.2, 
                 tournament_size=3,
                 # Initialization
                 init_method='random', 
                 param_ranges=None, 
                 mean=0, 
                 std_dev=1, 
                 custom_params=None):
        """
        Initializes the Genetic Algorithm with the specified parameters.

        Parameters:
        - generations (int): Number of generations in the optimization process.
        - population_size (int): Size of the population in each generation.
        - mutation_rate (float): Probability of mutation for an individual's parameters.
        - list_of_params (list): List of parameters to be optimized -> the solution-representation without fitness.
        - initial_solutions (list, optional): List of predefined solutions to start the optimization process. Defaults to an empty list.
            -> It is suggested to use the 'add_initial_solution' method instead.
        - parallel (bool, optional): Flag to enable parallel fitness calculation. Defaults to True.
        - should_print (bool, optional): Flag to control printing during optimization. Defaults to True.

        Mutation Parameters:
        - mutation_method (str, optional): Mutation method to use ('random', 'gaussian', 'uniform', or 'discrete'). Defaults to 'random'.
        - gaussian_std_dev (float, optional): Standard deviation for Gaussian mutation. Used when mutation_method is 'gaussian'.
        - uniform_min (float, optional): Minimum value for uniform mutation. Used when mutation_method is 'uniform'.
        - uniform_max (float, optional): Maximum value for uniform mutation. Used when mutation_method is 'uniform'.
        - discrete_choices (list, optional): List of discrete choices for mutation. Used when mutation_method is 'discrete'.

        Crossover Parameters:
        - crossover_possiblity (float, optional): Possibility of a crossover, some individuals will just taken how they are to the next generation. Defaults to 0.5.
        - crossover_method (str, optional): Method for crossover ('uniform', 'single_point', etc.). Defaults to 'uniform'.
        - crossover_points (tuple, optional): Points used for crossover operations. Interpretation depends on the method.

        Selection Parameters:
        - selection_method (str, optional): Method for selection ('tournament', 'roulette', etc.). Defaults to 'tournament'.
        - k (float, optional): Proportion of individuals to retain in 'k-best' selection methods. Defaults to 0.2.
        - tournament_size (int, optional): Size of tournament in tournament selection. Defaults to 3.

        Initialization Parameters:
        - init_method (str, optional): Method for initializing the population ('random', 'normal', etc.). Defaults to 'random'.
        - param_ranges (dict, optional): Parameter ranges for each param for random initialization. Format: {param_name: (min, max)}.
        - mean (float, optional): Mean for Gaussian initialization. Defaults to 0.
        - std_dev (float, optional): Standard deviation for Gaussian initialization. Defaults to 1.
        - custom_params (list, optional): Custom parameters to initialize population. Used when init_method is 'custom'.

        Returns:
        - None
        """
        # General config
        self.generations = generations
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.params = list_of_params
        self.initial_solutions = initial_solutions
        self.parallel = parallel
        self.should_print = should_print
        self.log = ""

        # Mutation config
        self.mutation_method = mutation_method
        self.gaussian_std_dev = gaussian_std_dev
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max
        self.discrete_choices = discrete_choices

        # Crossover config
        self.crossover_possiblity = crossover_possiblity
        self.crossover_method = crossover_method
        self.crossover_points = crossover_points

        # Selection config
        self.selection_method = selection_method
        self.k = k
        self.tournament_size = tournament_size

        # Initialization config
        self.init_method = init_method
        self.param_ranges = param_ranges
        self.mean = mean
        self.std_dev = std_dev
        self.custom_params = custom_params


    def add_initial_solution(self, solution:dict):
        """
        Adds an initial solution to the population.
        To lead the the algorithm in the right direction.

        Parameters:
        - solution (dict): Dictionary representing an initial solution with hyperparameter values.

        Returns:
        - None
        """
        self.initial_solutions += [[solution, float('-inf')]]

    def get_short_duration_representation(self, start, end):
        """
        Calculates and formats the duration between two time points.

        Parameters:
        - start (datetime): Start time.
        - end (datetime): End time.

        Returns:
        - str: Formatted duration string in the format "xD xH xM xS".
        """
        duration = abs((start-end).total_seconds())
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        res = f"{int(days)}D {int(hours)}H {int(minutes)}M {int(seconds)}S"
        return res

    @abstractmethod
    def calculate_fitness(self, kwargs, params):
        """
        Abstract method to calculate fitness for an individual.

        Parameters:
        - kwargs: The input(s) which the user give at the call of the optimize method.
        - params (dict): Dictionary representing the parameters of an individual.

        Returns:
        - float: Fitness value for the individual. 
        (Important) The higher the value, the better. The lower the value, the worse is the solutin!
        """
        pass

    @abstractmethod
    def get_random_param_value(self, param_key):
        """
        Abstract method to get a random value for a parameter of one solution.
        -> One solution consist out of many parameters as a dictionary.

        Parameters:
        - param_key: Key identifying the parameter of the solution.

        Returns:
        - any: Random value for the specified parameter.
        """
        pass

    def init_population(self, population_size, init_method='random', param_ranges=None, mean=0, std_dev=1, custom_params=None):
        """
        Initializes the population with different initialization methods.

        Parameters:
        - population_size (int): Size of the population.
        - init_method (str): Initialization method: 'random', 'uniform', 'gaussian', or 'custom'.
        - param_ranges (dict): Dictionary of min and max values for parameters (required for 'uniform' and 'gaussian').
        - mean (float): Mean for Gaussian distribution (used in 'gaussian' method).
        - std_dev (float): Standard deviation for Gaussian distribution (used in 'gaussian' method).
        - custom_params (dict): Dictionary of custom initial values for parameters (required for 'custom').

        Returns:
        - list: List of individuals, each represented as [params, fitness].
        """
        new_population = []
        
        for _ in range(population_size):
            params = dict()
            
            if init_method == 'random':
                # Random initialization (original standard method)
                for param in self.params:
                    params[param] = self.get_random_param_value(param)

            elif init_method == 'uniform':
                # Uniform initialization: sample within a given range for each parameter
                if param_ranges is None:
                    raise ValueError("param_ranges must be provided for 'uniform' initialization.")
                for param in self.params:
                    min_val, max_val = param_ranges.get(param, (0, 1))
                    params[param] = random.uniform(min_val, max_val)

            elif init_method == 'gaussian':
                # Gaussian initialization: sample from a normal distribution
                if param_ranges is None:
                    raise ValueError("param_ranges must be provided for 'gaussian' initialization.")
                for param in self.params:
                    min_val, max_val = param_ranges.get(param, (0, 1))
                    # Ensure the generated value is within the parameter range
                    while True:
                        value = np.random.normal(mean, std_dev)
                        if min_val <= value <= max_val:
                            params[param] = value
                            break

            elif init_method == 'custom':
                # Custom initialization: use pre-defined custom values
                if custom_params is None:
                    raise ValueError("custom_params must be provided for 'custom' initialization.")
                for param in self.params:
                    params[param] = custom_params.get(param, self.get_random_param_value(param))

            else:
                raise ValueError(f"Unknown initialization method: {init_method}")
            
            new_population.append([params, float('-inf')])
        
        return new_population

    def select_individuals(self, population, population_size, method='elitism', k=0.2, tournament_size=3):
        """
        Selects individuals from the population based on the chosen selection method.

        Parameters:
        - population (list): List of individuals, each as [params, fitness].
        - population_size (int): Total number of individuals in the population.
        - method (str): Selection method: 'elitism', 'roulette', or 'tournament'.
        - k (float): Fraction of individuals to select (for elitism and roulette).
        - tournament_size (int): Number of individuals per tournament (for tournament method).

        Returns:
        - list: Selected individuals from the population.
        """
        population = copy.deepcopy(population)
        
        if method == 'elitism':    # (original standard method)
            population = sorted(population, key=lambda x: x[1], reverse=True)
            selected_population = population[:int(k * population_size)]

        elif method == 'roulette':
            total_fitness = sum(ind[1] for ind in population)
            if total_fitness == 0:
                selected_population = random.sample(population, int(k * population_size))
            else:
                selection_probs = [ind[1] / total_fitness for ind in population]
                selected_population = random.choices(population, weights=selection_probs, k=int(k * population_size))

        elif method == 'tournament':
            selected_population = []
            for _ in range(int(k * population_size)):
                competitors = random.sample(population, tournament_size)
                winner = max(competitors, key=lambda x: x[1])
                selected_population.append(winner)

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return selected_population

    def crossover(self, parents, possibility=0.5, crossover_method='uniform', crossover_points=(0, 1)):
        """
        Performs crossover to create offspring using different crossover methods.

        Parameters:
        - parents (list): List of parent individuals, each represented as [params, fitness].
        - possibility (float, optional): Probability of crossover. Defaults to 0.5.
        - crossover_method (str): Crossover method: 'one_point', 'two_point', or 'uniform'.
        - crossover_points (tuple): Points used for 'one_point' or 'two_point' crossover.

        Returns:
        - list: Offspring population.
        """
        parents = copy.deepcopy(parents)
        offspring_population = []
        end_idx = len(parents)
        last_one = None
        if end_idx % 2 != 0:
            end_idx -= 1
            last_one = end_idx

        for parent_idx in range(0, end_idx, 2):
            parent_1 = parents[parent_idx]
            parent_2 = parents[parent_idx + 1]

            if random.random() > possibility:
                if crossover_method == 'one_point':
                    # One-Point Crossover: Select a point and swap before and after
                    crossover_point = random.randint(0, len(parent_1[0]) - 1)
                    child_params = {key: parent_1[0][key] if list(parent_1[0].keys()).index(key) < crossover_point else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

                elif crossover_method == 'two_point':
                    # Two-Point Crossover: Select two points and swap between them
                    point1, point2 = sorted(crossover_points)
                    child_params = {key: parent_1[0][key] if point1 <= list(parent_1[0].keys()).index(key) < point2 else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

                elif crossover_method == 'uniform':
                    # Uniform Crossover: Each parameter has an equal chance to come from either parent
                    child_params = {key: parent_1[0][key] if random.random() < 0.5 else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

                else:
                    raise ValueError(f"Unknown crossover method: {crossover_method}")
            else:
                offspring_population.append(parent_1)

            # Second parent -> child or him/herself
            if random.random() > possibility:
                if crossover_method == 'one_point':
                    crossover_point = random.randint(0, len(parent_1[0]) - 1)
                    child_params = {key: parent_1[0][key] if list(parent_1[0].keys()).index(key) < crossover_point else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

                elif crossover_method == 'two_point':
                    point1, point2 = sorted(crossover_points)
                    child_params = {key: parent_1[0][key] if point1 <= list(parent_1[0].keys()).index(key) < point2 else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

                elif crossover_method == 'uniform':
                    child_params = {key: parent_1[0][key] if random.random() < 0.5 else parent_2[0][key]
                                    for key in parent_1[0]}
                    offspring_population.append([child_params, float('-inf')])

            else:
                offspring_population.append(parent_2)

        # Add parent without a pair if population size is odd
        if last_one is not None:
            offspring_population.append(parents[last_one])

        return offspring_population

    def mutate_population(self, population, mutation_rate, population_size, 
                          mutation_method='random', 
                          gaussian_std_dev=None, 
                          uniform_min=None, uniform_max=None, 
                          discrete_choices=None):
        """
        Mutates the population based on the mutation rate using different mutation methods.

        Parameters:
        - population (list): List of individuals, each represented as [params, fitness].
        - mutation_rate (float): Probability of mutation for an individual's hyperparameters.
        - population_size (int): Amount of individuals/solution-fitness pairs.
        - mutation_method (str): Mutation method: 'random', 'gaussian', 'uniform', or 'discrete'.
        - gaussian_std_dev (float, optional): Standard deviation for gaussian mutation.
        - uniform_min (float, optional): Minimum value for uniform mutation.
        - uniform_max (float, optional): Maximum value for uniform mutation.
        - discrete_choices (list, optional): List of discrete choices for mutation.

        Returns:
        - list: Mutated population.
        """
        population = copy.deepcopy(population)
        
        for random_individual_idx in range(len(population)):
            if random.random() <= mutation_rate:
                for param in self.params:
                    if mutation_method == 'random':
                        # Random Mutation: Replace with random value from get_random_param_value
                        if random.random() > 0.5:
                            population[random_individual_idx][0][param] = self.get_random_param_value(param)

                    elif mutation_method == 'gaussian':
                        # Gaussian Mutation: Add Gaussian noise to the parameter value
                        if gaussian_std_dev is None:
                            raise ValueError("Missing 'gaussian_std_dev' for gaussian mutation")
                        noise = random.gauss(0, gaussian_std_dev)
                        population[random_individual_idx][0][param] += noise

                    elif mutation_method == 'uniform':
                        # Uniform Mutation: Replace the parameter with a random value within a specific range
                        if uniform_min is None or uniform_max is None:
                            raise ValueError("Missing 'uniform_min' or 'uniform_max' for uniform mutation")
                        population[random_individual_idx][0][param] = random.uniform(uniform_min, uniform_max)

                    elif mutation_method == 'discrete':
                        # Discrete Mutation: Change the parameter by picking a random discrete value
                        if discrete_choices is None:
                            raise ValueError("Missing 'discrete_choices' for discrete mutation")
                        population[random_individual_idx][0][param] = random.choice(discrete_choices)

                    else:
                        raise ValueError(f"Unknown mutation method: {mutation_method}")

                # Reset fitness after mutation
                population[random_individual_idx][1] = float('-inf')

        return population

    def add_newcomers(self, population, population_size):
        """
        'Fills' the population with new random solutions/individuals so that the population_size stays the same.

        Parameters:
        - population (list): List of individuals, each represented as [params, fitness].
        - population_size (int): Size of the population.

        Returns:
        - list: Updated population with new individuals.
        """
        population = copy.deepcopy(population)
        needed_individuals = int(population_size - len(population))
        if needed_individuals > 0:
            new_individuals = self.init_population(needed_individuals)
            population += new_individuals
        return population

    def calculate_fitness_wrapper(self, kwargs, cur_individual, idx, len_population):
        """
        Wrapper for parallel fitness calculation.

        Parameters:
        - kwargs: Additional keyword arguments for the fitness calculation.
        - cur_individual (dict): Parameters of the solution
        - idx (int): Index of the current individual.
        - len_population (int): Total number of individuals in the population.

        Returns:
        - tuple: Tuple containing index and (individual, fitness).
        """
        fitness = self.calculate_fitness(kwargs, cur_individual)
        return idx, (cur_individual, fitness)

    def optimize(self, **kwargs):
        """
        Searching for the best solution using a Genetic Algorithm.

        Parameters:
        - kwargs: All input values needed in your calculate_fitness.

        Returns:
        - tuple: Tuple containing the best hyperparameters, best fitness value, and log of the process.
        """
        population_size = self.population_size
        mutation_rate = self.mutation_rate
        generations = self.generations

        best_params = None
        best_fitness = float('-inf')

        start = dt.now()
        start_str = f"Hyperparameter Optimization with Genetic Algorithm\n    - start: {start.strftime('%Y-%m-%d %H:%M OClock')}\
        \n    - generations: {generations}\n    - population size: {population_size}\n    - mutation rate: {mutation_rate}"
        if self.should_print:
            print(start_str)
        self.log += start_str
        population = self.initial_solutions

        for generation in range(generations):
            start_generation = dt.now()
            start_gen_str = f"\n\n{'#'*16}\n {generation+1}. Generation ({generation+1}/{generations})\n    -> {start_generation.strftime('%Y-%m-%d %H:%M OClock')}"
            if self.should_print:
                print(start_gen_str)
            self.log += f"\n\n{start_gen_str}"

            # generate random population with hyperparamters
            if generation == 0:
                new_population_size = population_size - len(population)
                new_pop_str = f"    -> init {new_population_size} new population... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                new_pop_str += f"\n            = Uses {len(population)} predefined solutions!"
                if self.should_print:
                    print(new_pop_str)
                self.log += f"\n{new_pop_str}"
                population += self.init_population(population_size=new_population_size,
                                                   init_method=self.init_method,
                                                   param_ranges=self.param_ranges,
                                                   mean=self.mean,
                                                   std_dev=self.std_dev,
                                                   custom_params=self.custom_params)

            # calculate the fitness of the population
            if self.parallel:
                calc_individual_str = f"    -> parallel calc of all individuals fitness... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(calc_individual_str)
                self.log += f"\n{calc_individual_str}"
                results = Parallel(n_jobs=-1)(
                    delayed(self.calculate_fitness_wrapper)(kwargs, cur_individual[0], idx, len(population))
                    for idx, cur_individual in enumerate(population)
                )

                    # update results
                for idx, (individual, fitness) in results:
                    population[idx][0] = individual    # just for savety, should be fine
                    population[idx][1] = fitness
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_params = individual
            else:
                for idx, cur_individual in enumerate(population):
                    calc_individual_str = f"    -> calc individual fitness {idx+1}/{len(population)}... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                    if self.should_print:
                        print(calc_individual_str)
                    self.log += f"\n{calc_individual_str}"
                    cur_individual[1] = self.calculate_fitness(kwargs, cur_individual[0])

                    if cur_individual[1] > best_fitness:
                        best_fitness = cur_individual[1]
                        best_params = cur_individual[0]

            # skip last generation
            if generation != generations-1:

                # select best individuals as parents for the next generation
                select_k_best_str = f"    -> select best individuals... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(select_k_best_str)
                self.log += f"\n{select_k_best_str}"
                parents = self.select_individuals(population=population, 
                                                  population_size=population_size,
                                                  method=self.selection_method,
                                                  k=self.k,
                                                  tournament_size=self.tournament_size)

                # crossover - combine parents to create the 80% of the new generation
                crossover_str = f"    -> making gene crossovers... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(crossover_str)
                self.log += f"\n{crossover_str}"
                population = self.crossover(parents=parents,
                                            possibility=self.crossover_possiblity,
                                            crossover_method=self.crossover_method,
                                            crossover_points=self.crossover_points)

                # mutation
                mutation_str = f"    -> making mutations... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(mutation_str)
                self.log += f"\n{mutation_str}"
                population = self.mutate_population(population=population, 
                                                    mutation_rate=self.mutation_rate, 
                                                    population_size=self.population_size, 
                                                    mutation_method=self.mutation_method,
                                                    gaussian_std_dev=self.gaussian_std_dev,
                                                    uniform_min=self.uniform_min,
                                                    uniform_max=self.uniform_max,
                                                    discrete_choices=self.discrete_choices)

                # add new individuals
                new_ind_str = f"    -> create new individuals... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(new_ind_str)
                self.log += f"\n{new_ind_str}"
                population = self.add_newcomers(population, population_size)

            end = dt.now()
            end_str = f"Best Fitness: {best_fitness}\nWith Params: {best_params}"
            end_str += f"\n\nGeneration-Duration:{self.get_short_duration_representation(start_generation, end)}\nTotal-Duration:{self.get_short_duration_representation(start, end)}"
            self.log += f"\n\n{end_str}"
            if self.should_print:
                print(end_str)

        return best_params, best_fitness, self.log


