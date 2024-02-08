import random
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, solution_length, population_size, elite_count, num_generations, mutation_rate):
        self.solution_length = solution_length
        self.population_size = population_size
        self.elite_count = elite_count
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate

    @staticmethod
    def create_individual(length):
        return ''.join(random.choice('01') for _ in range(length))

    def generate_population(self):
        return [self.create_individual(self.solution_length) for _ in range(self.population_size)]

    @staticmethod
    def mutate(individual, mutation_rate):
        return ''.join('1' if bit == '0' and random.random() < mutation_rate else
                       '0' if bit == '1' and random.random() < mutation_rate else
                       bit for bit in individual)

    @staticmethod
    def crossover(parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    @staticmethod
    def roulette_selection(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        selection_probs = [fitness / total_fitness for fitness in fitness_scores]
        return random.choices(population, weights=selection_probs, k=len(population))

    def elitism(self, population, fitness_scores):
        sorted_population = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
        elite_individuals = [individual for individual, _ in sorted_population[:self.elite_count]]
        return elite_individuals

    @staticmethod
    def fitness_one_max(problem_instance):
        return sum(bit == '1' for bit in problem_instance)

    @staticmethod
    def fitness_target_string(problem_instance, target):
        return sum(s == t for s, t in zip(problem_instance, target))

    @staticmethod
    def fitness_deceptive(problem_instance):
        count_ones = sum(bit == '1' for bit in problem_instance)
        return 2 * len(problem_instance) if count_ones == 0 else count_ones

    def run_genetic_algorithm(self, fitness_func, *additional_args):
        population = self.generate_population()
        avg_fitness_history = []

        for _ in range(self.num_generations):
            fitness_scores = [fitness_func(ind, *additional_args) if additional_args else fitness_func(ind) for ind in population]
            elite = self.elitism(population, fitness_scores)
            selected_population = self.roulette_selection(population, fitness_scores)
            selected_population = selected_population[:-self.elite_count]

            new_population = elite
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected_population, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1, self.mutation_rate), self.mutate(child2, self.mutation_rate)])

            population = new_population[:self.population_size]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            avg_fitness_history.append(avg_fitness)

        return avg_fitness_history

    def plot_graph(self, history, problem_name):
        plt.plot(history)
        plt.ylim(0, 30)
        plt.title(f'{problem_name}')
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.grid(True)
        plt.savefig(f'{problem_name}_genetic_algorithm_results.png', bbox_inches='tight')
        plt.show()

# Running for each problem and plotting individual graphs
# 1.1 One-max Problem
ga_one_max = GeneticAlgorithm(30, 1000, 2, 100, 0.01)
one_max_history = ga_one_max.run_genetic_algorithm(GeneticAlgorithm.fitness_one_max)
ga_one_max.plot_graph(one_max_history, 'One-max Problem')

# 1.2 Target String Problem
target_string = "00101011001101001001"
ga_target_string = GeneticAlgorithm(30, 1000, 2, 100, 0.01)
target_string_history = ga_target_string.run_genetic_algorithm(GeneticAlgorithm.fitness_target_string, target_string)
ga_target_string.plot_graph(target_string_history, 'Target String Problem')

# 1.3 Deceptive Landscape
ga_deceptive = GeneticAlgorithm(30, 1000, 2, 100, 0.01)
deceptive_history = ga_deceptive.run_genetic_algorithm(GeneticAlgorithm.fitness_deceptive)
ga_deceptive.plot_graph(deceptive_history, 'Deceptive Landscape')
