# imports
import random
import copy
from datetime import datetime as dt
from joblib import Parallel, delayed
import enum

def get_random(*args):
    """
    A helper function to create the get_random_param_value method.
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

class GA():

    def __init__(self, generations, population_size, mutation_rate, list_of_params, parallel=True, should_print=True):
        self.params = list_of_params
        self.should_print = should_print
        self.log = ""
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.parallel = parallel

    def get_short_duration_representation(self, start, end):
        duration = abs((start-end).total_seconds())
        minutes, seconds = divmod(duration, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)
        res = f"{int(days)}D {int(hours)}H {int(minutes)}M {int(seconds)}S"
        return res

    def calculate_fitness(self, kwargs, params):
        pass

    def get_random_param_value(self, param_key):
        pass

    def init_population(self, population_size):
        new_population = []
        for _ in range(population_size):
            params = dict()
            for param in self.params:
                params[param] = self.get_random_param_value(param)

            new_population += [[params, float('-inf')]]
        return new_population

    def select_individuals(self, population, population_size):
        population = copy.deepcopy(population)
        population = sorted(population, key=lambda x: x[1], reverse=True)
        # 20% best are the new parents for the next generation
        selected_population = population[:int(0.2 * population_size)]
        #selected_population = random.choices(population, weights=fitnesses, k=int(0.2 * len(population)))
        return selected_population

    def crossover(self, parents, possiblity=0.5):
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
        population = copy.deepcopy(population)
        for random_individual_idx in range(len(population)):
            if random.random() <= mutation_rate:
                for param in self.params:
                    if random.random() > 0.5:
                        population[random_individual_idx][0][param] = self.get_random_param_value(param)

                population[random_individual_idx][1] = float('-inf')
            
        return population

    def add_newcomers(self, population, population_size):
        population = copy.deepcopy(population)
        needed_individuals = int(population_size - len(population))
        if needed_individuals > 0:
            new_individuals = self.init_population(needed_individuals)
            population += new_individuals
        return population

    def calculate_fitness_wrapper(self, kwargs, cur_individual, idx, len_population):
        # calc_individual_str = f"    -> calc individual fitness {idx+1}/{len_population}... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
        # if self.should_print:
        #     print(calc_individual_str)
        # self.log += f"\n{calc_individual_str}"
        fitness = self.calculate_fitness(kwargs, cur_individual)
        return idx, (cur_individual, fitness)

    def optimize(self, **kwargs):
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
        population = []

        for generation in range(generations):
            start_generation = dt.now()
            start_gen_str = f"\n\n{'#'*16}\n {generation+1}. Generation ({generation+1}/{generations})\n    -> {start_generation.strftime('%Y-%m-%d %H:%M OClock')}"
            if self.should_print:
                print(start_gen_str)
            self.log += f"\n\n{start_gen_str}"

            # generate random population with hyperparamters
            if generation == 0:
                new_pop_str = f"    -> init new population... ({dt.now().strftime('%Y-%m-%d %H:%M OClock')})"
                if self.should_print:
                    print(new_pop_str)
                self.log += f"\n{new_pop_str}"
                population = self.init_population(population_size)

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


