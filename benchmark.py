import numpy as np
import time
from engine import Engine
from utils import manual_test

# m, n, and d are constant integers we can define before we initialize matrices.
# m - number of candidates in database, n - number of queries in queries (Q), d - number of features for each candidate/query.
m, n, d = 10000, 10, 32

# initializing matrices randomly.
D = np.random.randn(m, d)
Q = np.random.randn(n, d)

# initializing 2 engines (1 for l2, 1 for cosine)
metric_1 = 'l2'
metric_2 = 'cosine'
engine_1 = Engine(metric_1)
engine_1.add_to_index(D)
engine_2 = Engine(metric_2)
engine_2.add_to_index(D)

n_query = np.random.randint(0,n)

# records the time of the start of the benchmark session. 
# Every time.time() from now on will return the time when the search ends for each test
start_time = time.perf_counter()

engine_1.search(Q[n_query].reshape(1,-1), 1)
engine_1_end_time = time.perf_counter()

engine_2.search(Q[n_query].reshape(1,-1), 1)
engine_2_end_time = time.perf_counter()

manual_test(D, Q, metric_1, n_query)
manual_test_1_end_time = time.perf_counter()

manual_test(D, Q, metric_2, n_query)
manual_test_2_end_time = time.perf_counter()


# calculating time it took for each search session
engine_1_time = engine_1_end_time - start_time
engine_2_time = engine_2_end_time - engine_1_end_time
manual_test_1_time = manual_test_1_end_time - engine_2_end_time
manual_test_2_time = manual_test_2_end_time - manual_test_1_end_time


# calculates how much faster it is to use vectorized algorithm
l2_ratio = manual_test_1_time / engine_1_time
cosine_ratio = manual_test_2_time / engine_2_time

print(l2_ratio)
print(cosine_ratio)

