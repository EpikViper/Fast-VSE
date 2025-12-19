import numpy as np
from engine import Engine
from utils import manual_test

m, n, d = 1000, 10, 32

# specified float64 for high precision when testing
D = np.random.randint(0,10, size=(m,d)).astype(np.float64)
Q = np.random.randint(0,10, size=(n,d)).astype(np.float64)

n_query = np.random.randint(0,n)

# Can also be l2.
metric = 'cosine'

engine = Engine(metric)
engine.add_to_index(D)

# to unwrap the row index from double bracakets, we use .item()
engine_result = engine.search(Q[n_query].reshape(1,-1), 1).item()

manual_result = manual_test(D, Q, metric, n_query)

# calculates distance between n-th query and the m-th candidate in D, where m is the result of our calculation
engine_distance = np.linalg.norm(Q[n_query] - D[engine_result])
manual_distance = np.linalg.norm(Q[n_query] - D[manual_result])


# prints distance calculated both manually and using our optimized engine.
print(engine_distance)
print(manual_distance)