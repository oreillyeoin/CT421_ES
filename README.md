# CT421_ES
CT421 Project 1: Evolutionary Search: Description of Code Structure
The evolutionary search script defines methods for each operation to keep complexity down and make the code clear and readable. It then defines a larger method for the execution of the algorithm, taking an argument to define which fitness metric to use, allowing it to be repurposed for each search problem. This method returns an array of the average fitness across all generations, allowing the script to plot graphs for each problem.


Variables:
- solution_length: Length of the binary string representing a solution.
- population_size: Size of the population in each generation.
- elite_count: Number of top individuals preserved in each generation.
- num_generations: Number of generations for the algorithm to run.
- mutation_rate: Probability of mutation for each bit in the string.


Methods:
- create_individual: This method is used to create each individual, generating a random binary string of specified length. 

- generate_population: This method then calls this create_individual method within a loop of population_size (the desired size of the population) to create the initial population.

- mutate: This method performs a mutation on an individual string based on mutation rate, for each bit in the string, the bit is flipped based on the mutation_rate. A new string is then returned after all the bits have been looped through.

- crossover: This method implements one-point crossover between two parent strings.
It randomly selects a crossover point and combines the substrings of parents before and after the crossover point to create two new children.

- roulette_selection: This method performs roulette wheel selection to choose individuals from the population based on their fitness scores. It calculates selection probabilities based on fitness scores and uses random.choices to select individuals.

- elitism: This method selects the top individuals based on their fitness scores to be preserved in the next generation. It sorts the population based on fitness scores and extracts the top individuals, preserving them.

- fitness_one_max: This method calculates the fitness for the One-Max problem, which is the count of '1' bits in the binary string.

- fitness_target_string: This method calculates the fitness for the Target String problem, which is the count of matching bits between the problem instance and the target string.

- fitness_deceptive: This method calculates the fitness for a deceptive landscape, where it returns double the length of the problem instance if there are no '1' bits, otherwise, it returns the count of '1' bits.

- run_genetic_algorithm: This method orchestrates the entire genetic algorithm, calling all the above methods when necessary. It initializes a population, evolves it through generations using selection, crossover, mutation, and elitism, and records the average fitness over generations. The loop iterates for num_generations, updating the population and fitness scores accordingly. It returns the average fitness score throughout all generations in the form of an array, which can be used for plotting graphs afterwards.


Running the script:

The run_genetic_algorithm method is called for each individual search problem, supplying the fitness method (e.g. fitness_one_max) as an argument to determine which search problem is being ran. 
