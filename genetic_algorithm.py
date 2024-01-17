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

    def __init__(self, generations, population_size, mutation_rate, list_of_params, initial_solutions=list(), parallel=True, should_print=True):
        """
        Initializes the Genetic Algorithm with the specified parameters.

        Parameters:
        - generations (int): Number of generations in the optimization process.
        - population_size (int): Size of the population in each generation.
        - mutation_rate (float): Probability of mutation for an individual's parameters.
        - list_of_params (list): List of parameters to be optimized -> the solution-representation without fitness.
        - initial_solutions (list, optional): List of predefined solutions to start the optimization process. Defaults to an empty list.
            -> It is suggested to use the 'add_initial_solution'-method instead.
        - parallel (bool, optional): Flag to enable parallel fitness calculation. Defaults to True.
        - should_print (bool, optional): Flag to control printing during optimization. Defaults to True.

        Returns:
        - None

        """
        self.params = list_of_params
        self.should_print = should_print
        self.log = ""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.parallel = parallel
        self.initial_solutions = initial_solutions

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

    def init_population(self, population_size):
        """
        Initializes the population with random solutions and -inf fitness.

        Parameters:
        - population_size (int): Size of the population.

        Returns:
        - list: List of individuals, each represented as [params, fitness].
        """
        new_population = []
        for _ in range(population_size):
            params = dict()
            for param in self.params:
                params[param] = self.get_random_param_value(param)

            new_population += [[params, float('-inf')]]
        return new_population

    def select_individuals(self, population, population_size):
        """
        Selects the best individuals from the population.

        Parameters:
        - population (list): List of individuals, each represented as [params, fitness].
        - population_size (int): Amount of individuals/solution-fitness pairs.

        Returns:
        - list: Selected individuals from the population.
        """
        population = copy.deepcopy(population)
        population = sorted(population, key=lambda x: x[1], reverse=True)
        # 20% best are the new parents for the next generation
        selected_population = population[:int(0.2 * population_size)]
        return selected_population

    def crossover(self, parents, possiblity=0.5):
        """
        Performs crossover to create offspring.

        Parameters:
        - parents (list): List of parent individuals, each represented as [params, fitness].
        - possibility (float, optional): Probability of crossover. Defaults to 0.5.

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
            parent_2 = parents[parent_idx+1]

            # first parent -> child or him/herself
            if random.random() > possiblity:
                child_params = {key: parent_1[0][key] if random.random() < 0.5 else parent_2[0][key] for key in parent_1[0]}
                offspring_population += [[child_params, float('-inf')]]
            else:
                offspring_population += [parent_1]

            # second parent -> child or him/herself
            if random.random() > possiblity:
                child_params = {key: parent_1[0][key] if random.random() < 0.5 else parent_2[0][key] for key in parent_1[0]}
                offspring_population += [[child_params, float('-inf')]]
            else:
                offspring_population += [parent_2]

        # add parent without a husband
        if type(last_one) != type(None):
            offspring_population += [parents[last_one]]

        return offspring_population

    def mutate_population(self, population, mutation_rate, population_size):
        """
        Mutates the population based on the mutation rate.

        Parameters:
        - population (list): List of individuals, each represented as [params, fitness].
        - mutation_rate (float): Probability of mutation for an individual's hyperparameters.
        - population_size (int): Amount of individuals/solution-fitness pairs.

        Returns:
        - list: Mutated population.
        """
        population = copy.deepcopy(population)
        for random_individual_idx in range(len(population)):
            if random.random() <= mutation_rate:
                for param in self.params:
                    if random.random() > 0.5:
                        population[random_individual_idx][0][param] = self.get_random_param_value(param)

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
                population += self.init_population(new_population_size)

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
                parents = self.select_individuals(population, population_size)

                # crossover - combine parents to create the 80% of the new generation
                crossover_str = f"    -> making gene crossovers... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(crossover_str)
                self.log += f"\n{crossover_str}"
                population = self.crossover(parents)

                # mutation
                mutation_str = f"    -> making mutations... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(mutation_str)
                self.log += f"\n{mutation_str}"
                population = self.mutate_population(population, mutation_rate, population_size)

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


