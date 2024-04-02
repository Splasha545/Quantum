import numpy as np
from itertools import product

def generate_binary_variables(n, m):
    # Generate all combinations of binary variables of size n*m
    bin_vars = list(product([0, 1], repeat = n*m))
    # Reshape the list of variables into a 2D array for easy addressing
    return [np.array(i).reshape(n,m) for i in bin_vars]

def compute_arithmetic_function(x, adj_matrix1, adj_matrix2, n, m):
    """
    Computes the arithmetic function (CUBO) based on the given parameters (pattern and target graphs).

    Parameters:
    x (list): A 2D list representing the mapping between vertices of two graphs.
    adj_matrix1 (list): A 2D list representing the adjacency matrix of the first graph.
    adj_matrix2 (list): A 2D list representing the adjacency matrix of the second graph.
    n (int): The number of vertices in the first graph.
    m (int): The number of vertices in the second graph.

    Returns:
    float: The computed arithmetic function value.

    """

    # OBJECTIVE FUNCTION (MINIMIZATION):
    sum = 0
    for i in range(n):
        for u in range(m):
            sum += -1*(x[i][u])
    
    # CONSTRAINT 1 (CHECKS THAT EACH VERTEX IN G1 IS MAPPED TO AT MOST ONE VERTEX IN G2):
    sum2 = 0
    for v in range(m):
        inner = 0
        for i in range(n):
            inner += x[i][v]
        sum2 += pow((0.5 - inner), 2) - 0.25
    sum += sum2

    sum2 = 0
    # CONSTRAINT 2 (CHECKS THAT NUMBER OF OUTGOING EDGES OF EACH VERTEX U IS EQUAL TO OUTGOING EDGES OF EACH VERTEX I):
    for u in range(m):
        for i in range(n):
            if (x[i][u] == 1):
                # First term
                fst = 0
                snd = 0
                for j in range(n):
                    fst = 0
                    snd = 0
                    if adj_matrix1[i][j] == 1:
                        fst += 1
                    # Second term:
                    for v in range(m):
                        if (adj_matrix2[u][v] == 1):
                            snd += x[j][v]
                
                # Their difference:
                    sum2 += n*pow(snd-fst, 2)
    sum += sum2

    sum2 = 0
    # CONSTRAINT 3 (CHECKS THAT NUMBER OF INCOMING EDGES OF EACH VERTEX U IS EQUAL TO OUTGOING EDGES OF EACH VERTEX I):
    for u in range(m):
        for i in range(n):
            if (x[i][u] == 1):
                # First term
                fst = 0
                for j in range(n):
                    fst = 0
                    snd = 0
                    if adj_matrix1[j][i] == 1:
                        fst += 1
                    # Second term:
                    for v in range(m):
                        if (adj_matrix2[v][u] == 1):
                            snd += x[j][v]
                
                # Their difference:
                    sum2 += n*pow(snd-fst, 2)
    sum += sum2

    return sum

def main(adj_matrix1, adj_matrix2):
    """
    Finds the best combination of binary variables that minimizes the result of the CUBO cost function.

    Args:
        adj_matrix1 (list): The adjacency matrix 1.
        adj_matrix2 (list): The adjacency matrix 2.

    """
    n = len(adj_matrix1)
    m = len(adj_matrix2)
    binary_variables = generate_binary_variables(n, m)

    min = 1000
    best = None
    for variables in binary_variables:
        result = compute_arithmetic_function(variables, adj_matrix1, adj_matrix2, n, m)
        if result < min:
            min = result
            best = variables
    print(f"Final result:\n Minimum: {min}\n Best combination:\n {best}")

def compute(adj_matrix1, adj_matrix2):
    """
    Computes the arithmetic function using the given adjacency matrices.

    Parameters:
    adj_matrix1 (list): The first adjacency matrix.
    adj_matrix2 (list): The second adjacency matrix.
    """
    
    n = len(adj_matrix1)
    m = len(adj_matrix2)

    # print(compute_arithmetic_function(
    #                             [[0, 0, 0, 0, 0, 1, 0],
    #                              [0, 0, 1, 1, 0, 0, 0],
    #                              [0, 0, 0, 0, 1, 0, 0]], adj_matrix1, adj_matrix2, n, m))
    # print(compute_arithmetic_function(
    #                             [[0, 0, 0, 1, 0, 1, 0],
    #                              [0, 0, 1, 0, 1, 0, 0],
    #                              [0, 1, 0, 0, 0, 0, 1]], adj_matrix1, adj_matrix2, n, m))
    # print(compute_arithmetic_function(
    #                             [[0, 0, 0, 0, 0, 1, 0],
    #                              [0, 0, 0, 0, 1, 0, 0],
    #                              [0, 1, 1, 1, 0, 0, 0]], adj_matrix1, adj_matrix2, n, m))
    # print(compute_arithmetic_function(
    #                             [[1, 0, 1, 0],
    #                              [0, 1, 0, 0],
    #                              [0, 0, 0, 1]], adj_matrix1, adj_matrix2, n, m))
    # print(compute_arithmetic_function(
    #                             [[1, 0, 0, 0, 0],
    #                              [0, 1, 0, 0, 0],
    #                              [0, 0, 0, 0, 1]], adj_matrix1, adj_matrix2, n, m))
    print(compute_arithmetic_function(
                                [[1, 0, 0, 0],
                                 [0, 1, 0, 0]], adj_matrix1, adj_matrix2, n, m))
    

# Define your adjacency matrices here
adj_matrix1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
adj_matrix2 = np.array([[0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 0]])
adj_matrix1 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
adj_matrix2 = np.array([[0, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0],])
# adj_matrix1 = np.array([[0, 1], [0, 0]])
# adj_matrix2 = np.array([[0, 1, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 1],
#                         [0, 0, 0, 0]])
# adj_matrix1 = np.array([[0, 1], [0, 0]])
# adj_matrix2 = np.array([[0, 1, 0, 0],
#                         [0, 0, 0, 0],
#                         [0, 0, 0, 1],
#                         [0, 0, 0, 0]])
# adj_matrix1 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# adj_matrix2 = np.array([[0, 1, 0, 1],
#                        [0, 0, 0, 1],
#                        [0, 1, 0, 1],
#                        [0, 0, 0, 0]])
# adj_matrix1 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
# adj_matrix2 = np.array([[0, 1, 0, 0, 1],
#                         [0, 0, 1, 0, 1],
#                         [0, 0, 0, 1, 1],
#                         [1, 0, 0, 0, 1],
#                         [0, 0, 0, 0, 0]])
# adj_matrix1 = np.array([[0, 1], [0, 0]])
# adj_matrix2 = np.array([[0, 1, 0],
#                         [0, 0, 0],
#                         [0, 1, 0]])

#compute(adj_matrix1, adj_matrix2)
main(adj_matrix1, adj_matrix2)