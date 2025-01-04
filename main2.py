import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd


def generate_random_route(num_cities):
    route = list(range(num_cities))
    random.shuffle(route)
    return route

def calculate_route_cost(route, cost_matrix):
    cost = 0
    for i in range(len(route)):
        cost += cost_matrix[route[i - 1]][route[i]]
    return cost

# Оператори схрещування
def ordered_crossover(parent1, parent2):
    size = len(parent1)
    start_point, end_point = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[start_point:end_point] = parent1[start_point:end_point]

    p2_pointer = 0
    for i in range(size):
        if child[i] == -1:
            while parent2[p2_pointer] in child:
                p2_pointer += 1
            child[i] = parent2[p2_pointer]

    return child

def single_point_crossover(parent1, parent2):
    size = len(parent1)
    split_point = random.randint(1, len(parent1) - 1)
    child = [-1] * size
    child[:split_point] = parent1[:split_point]
    
    index_to_fill = split_point
    for city in parent2:
        if city not in child[:split_point]:
            child[index_to_fill] = city
            index_to_fill += 1

    return child

def uniform_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    
    for i in range(size):
        if random.random() < 0.5:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    seen = set()
    for i in range(size):
        if child[i] in seen:
            for val in parent1 + parent2:
                if val not in child and val not in seen:
                    child[i] = val
                    break
        seen.add(child[i])
    
    return child

# Оператори мутації
def swap_mutation(route):
    size = len(route)
    i, j = random.sample(range(size), 2)
    route[i], route[j] = route[j], route[i]
    return route

def inversion_mutation(route):
    start, end = sorted(random.sample(range(len(route)), 2))
    route[start:end] = route[start:end][::-1]
    return route

# Оператори локального покращення
def double_bridge_move(route, cost_matrix, attempts=1000):
    import random
    best_cost = calculate_route_cost(route, cost_matrix)
    best_route = route[:]
    n = len(route)

    for _ in range(attempts):
        # Вибираємо 4 точки розрізу
        splits = sorted(random.sample(range(1, n), 3))
        i, j, k = splits
        # Розрізаємо на 4 сегменти
        A = route[:i]
        B = route[i:j]
        C = route[j:k]
        D = route[k:]
        # Переставляємо сегменти
        new_route = A + C + B + D
        new_cost = calculate_route_cost(new_route, cost_matrix)
        if new_cost < best_cost:
            best_cost = new_cost
            best_route = new_route[:]

    return best_route

def ruin_and_recreate(route, cost_matrix, ruin_fraction=0.1):
    n = len(route)
    num_to_remove = int(n * ruin_fraction)
    removed_indices = sorted(random.sample(range(n), num_to_remove), reverse=True)

    removed_cities = []
    new_route = route[:]
    for idx in removed_indices:
        removed_cities.append(new_route.pop(idx))

    for city in removed_cities:
        best_pos = 0
        best_increase = float('inf')
        for i in range(len(new_route)+1):
            test_route = new_route[:i] + [city] + new_route[i:]
            increase = calculate_route_cost(test_route, cost_matrix)
            if increase < best_increase:
                best_increase = increase
                best_pos = i
        new_route.insert(best_pos, city)
    return new_route

def genetic_algorithm(cost_matrix, population_size, generations, crossover_operator, mutation_operator, local_optimization, mutation_probability):
    num_cities = len(cost_matrix)
    
    population = [generate_random_route(num_cities) for _ in range(population_size)]
    best_route = None
    best_cost = float('inf')
    
    for generation in range(generations):
        fitness = [calculate_route_cost(route, cost_matrix) for route in population]
        sorted_indices = np.argsort(fitness)
        population = [population[i] for i in sorted_indices]

        if fitness[sorted_indices[0]] < best_cost:
            best_route = population[0]
            best_cost = fitness[sorted_indices[0]]

        parent1, parent2 = random.sample(population[:population_size // 2], 2)
        child = crossover_operator(parent1, parent2)

        if random.random() < mutation_probability:
            child = mutation_operator(child)

        if local_optimization:
            child = local_optimization(child, cost_matrix)

        worst_index = population.index(population[-1])
        population[worst_index] = child

        if generation + 1 >= generations:
            fitness = [calculate_route_cost(route, cost_matrix) for route in population]
            population = [route for _, route in sorted(zip(fitness, population), key=lambda x: x[0])]

        iterations_list.append(generation)
        path_lengths.append(fitness[0])
        if generation % 50 == 0:
            table_rows.append([generation, fitness[0]])

    return best_route, best_cost


iterations_list = []
path_lengths = []
table_rows = []
output_csv = "genetic_algorithm_results_pandas.csv"

if __name__ == "__main__":
    cost_matrix = np.loadtxt("graph_matrix.csv", delimiter=",", dtype=int)

    population_size = 20
    generations = 50

    best_route, best_cost = genetic_algorithm(
        cost_matrix,
        population_size,
        generations,
        crossover_operator=uniform_crossover,
        mutation_operator=inversion_mutation,
        local_optimization=ruin_and_recreate,
        mutation_probability=0.3
    )

    table_rows_df = pd.DataFrame(table_rows)
    table_rows_df.to_csv(output_csv, index = False)

    print("Найкращий маршрут:", best_route)
    print("Вартість маршруту:", best_cost)

    plt.figure(figsize=(10, 10))
    plt.plot(iterations_list, path_lengths, marker='o', linestyle='-', color='b')
    plt.xlabel('Кількість ітерацій')
    plt.ylabel('Довжина шляху')
    plt.title('Залежність довжини шляху від ітерацій')
    plt.grid()
    plt.xlim(0, max(iterations_list) + 10)
    plt.ylim(0, max(path_lengths) + 10)
    plt.show()
