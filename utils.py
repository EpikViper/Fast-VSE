import numpy as np
from engine import Engine


# m, n, and d are constant integers we can define before we initialize matrices.
# m - number of candidates in database, n - number of queries in queries (Q), d - number of features for each candidate/query.
m, n, d = 100, 5, 32

# initializing matrices randomly.
D = np.random.randn(m, d)
Q = np.random.randn(n, d)

n_query = np.random.randint(0,n)


def brute_force(D, Q, metric, n_query):

    # initializes our engine.
    engine = Engine(metric)
    engine.add_to_index(D)
    # our engine expects a 2D matrix, where Q[n_query] is only 1D so we have to reshape it.
    top_choice = engine.search(Q[n_query].reshape(1,-1), 1)
    
    neighbor = manual_test(D, Q, metric, n_query)
    ## checks if our brute force calculation and our engine got the same results.
    return top_choice[0, 0] == neighbor


def manual_test(D, Q, metric, n_query):
    if metric == 'l2':
        neighbor = 0
        shortest_distance = np.sum((Q[n_query]- D[0])**2)
        for i in range(len(D)):
            distance = np.sum((Q[n_query]- D[i])**2)
            if distance < shortest_distance:
                shortest_distance = distance
                neighbor = i
        
    elif metric == 'cosine':
        neighbor = 0
        # large cosine means the vectors point in the directions that are close to each other
        largest_cosine = np.dot(Q[n_query], D[0]) / (np.linalg.norm(Q[n_query]) * np.linalg.norm(D[0]))
        for i in range(len(D)):
            cosine = np.dot(Q[n_query], D[i]) / (np.linalg.norm(Q[n_query]) * np.linalg.norm(D[i]))
            if cosine > largest_cosine:
                largest_cosine = cosine
                neighbor = i

    return neighbor