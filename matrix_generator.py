import random
import numpy as np


def generate_full_directed_graph_matrix(num_vertices, weight_range=(5, 150)):
    adjacency_matrix = np.zeros((num_vertices, num_vertices), dtype=int)

    for i in range(num_vertices):
        for j in range(num_vertices):
            if i != j:
                adjacency_matrix[i][j] = random.randint(*weight_range)
    
    return adjacency_matrix

num_vertices = 300
matrix = generate_full_directed_graph_matrix(num_vertices)

total_edges = np.count_nonzero(matrix)
print(f"Кількість вершин: {num_vertices}")
print(f"Кількість ребер: {total_edges}")

for i in range(num_vertices):
    print(matrix[i])

np.savetxt("graph_matrix.csv", matrix, delimiter=",", fmt='%d')