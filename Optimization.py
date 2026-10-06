import numpy as np
def perceptron_predict(inputs, weights, bias):
   total = np.dot(inputs, weights) + bias
   return 1 if total >= 0 else 0

def calculate_fitness(population, X, y):
    fitness_scores = []
    for individual in population:
        weights = individual[:-1]
        bias = individual[-1]
        predictions = [perceptron_predict(x, weights, bias) for x in X]
        accuracy = np.mean(np.array(predictions) == y)
        fitness_scores.append(accuracy)
    return fitness_scores
x = np.array([
[0, 0],
[0, 1],
[1, 0],
[1, 1]
])
y = np.array([0,0,0,1])
population_size = 10
num_features = 2
population = np.random.uniform(-1, 1, (population_size, num_features + 1))

print("Evolutionary weight optimization initialized successfully.")
print("\nPopulation:")
print(population)
print("\nFitness scores:")