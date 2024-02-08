import random
import sys
from collections import Counter, defaultdict
from statistics import mean
from matplotlib import pyplot as plt


class BinPackingAlgorithm:
    def __init__(self):
        self.population_size = 50
        self.max_gen = 300
        self.mutation_prob = 0.01

    def calculate_fitness(self, solution, items, capacity):
        # Calculate the fitness of a solution
        bins = Counter(solution)
        bin_weights = [sum(items[i] for i, bin_num in enumerate(solution) if bin_num == bin_) for bin_ in bins]

        penalty = sum(1 for weight in bin_weights if weight > capacity)
        fitness = len(bins) + penalty

        return fitness

    def tournament_selection(self, population, capacity, items):
        # Select parents using tournament selection
        parents = []
        for _ in range(2):
            contenders = random.sample(population, 3)
            contenders.sort(key=lambda ind: self.calculate_fitness(ind, items, capacity))
            parents.append(contenders[0])
        return parents

    def crossover(self, parent1, parent2):
        # Perform crossover to generate children
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, individual):
        # Perform mutation on an individual
        return [random.randint(1, self.population_size) if random.random() < self.mutation_prob else gene for gene in individual]

    def generate_children(self, population, capacity, items):
        # Generate children using crossover and mutation
        parent1, parent2 = self.tournament_selection(population, capacity, items)
        child1, child2 = self.crossover(parent1, parent2)
        child1 = self.mutate(child1)
        child2 = self.mutate(child2)
        return [child1, child2]

    def run_algorithm(self, instance):
        # Run the bin packing algorithm
        capacity = instance['capacity']
        items = []
        population = []

        # Extract item weights
        for weight, count in instance['items']:
            items.extend([weight] * count)

        # Generate an initial population of individuals
        for _ in range(self.population_size):
            individual = []
            for weight, count in instance['items']:
                individual.extend([random.randint(1, self.population_size)] * count)  # Assign to a random bin
            population.append(individual)

        # Randomize the order of the population
        for individual in population:
            random.shuffle(individual)

        best_fitness = sys.maxsize
        avg_fitness_history = []

        for i in range(self.max_gen):
            new_population = []

            # Generate children and add to the new population
            for _ in range(0, self.population_size, 2):
                new_population.extend(self.generate_children(population, capacity, items))

            # Sort the population based on fitness and select the top individuals
            population = sorted(new_population, key=lambda ind: self.calculate_fitness(ind, items, capacity))[:self.population_size]

            # Update the best fitness, average fitness, and history
            best_individual = population[0]
            best_fitness = min(best_fitness, self.calculate_fitness(best_individual, items, capacity))

            avg_fitness = mean(self.calculate_fitness(ind, items, capacity) for ind in population)
            avg_fitness_history.append(avg_fitness)

        return best_individual, avg_fitness_history


if __name__ == "__main__":
    bpp = BinPackingAlgorithm()

    # Read in the txt file
    with open('Binpacking.txt', 'r') as file:
        text = file.read()
        data = []
        lines = text.strip().split('\n')
        i = 0
        while i < len(lines):
            name = lines[i].strip().strip("'")
            i += 1
            m = int(lines[i].strip())
            i += 1
            capacity = int(lines[i].strip())
            i += 1
            items = []
            for _ in range(m):
                weight, count = map(int, lines[i].strip().split())
                items.append((weight, count))
                i += 1
            data.append({'name': name, 'capacity': capacity, 'items': items})

    # Run the algorithm for each bin problem
    for i in range(len(data)):
        best, avg_fitness_history = bpp.run_algorithm(data[i])
        plt.plot(avg_fitness_history, label=f"{data[i]['name']}")

        bins = defaultdict(list)
        for item_idx, bin_num in enumerate(best):
            item_weight = data[i]['items'][item_idx % len(data[i]['items'])][0]
            bins[bin_num].append(item_weight)  # Append the item's weight to the corresponding bin in the defaultdict

        print(f"{data[i]['name']}: \nNumber of Bins: {len(bins)}")
        for bin_num, items in bins.items():
            print(f"Bin {bin_num}: {items}")
        print("________________________________")

    # Plotting and saving the results
    plt.title("Average Best Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.savefig('binpacking_results.png')
    plt.show()
