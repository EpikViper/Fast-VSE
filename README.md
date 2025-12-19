# Fast VSE

A high-performance Python implementation of a VSE supporting **Euclidean Distance** and **Cosine Similarity**.  
This is a demonstration that NumPy vectorization can achieve significant speedups over iterative search methods.

## Method

Standard loops in Python are slow for matrix operations. By using the matrix expansion trick, we can calculate distances for thousands of vectors simultaneously.

Instead of looping through every pair, we use the formula:
$$\|A - B\|^2 = \|A\|^2 + \|B\|^2 - 2(A \cdot B)$$

## Benchmarks

Running the included `benchmark.py`, we see that:

**For m, n, d = 100, 10, 32:**

- L2 Ratio: ~4x Faster
- Cosine Ratio: ~17x Faster

**For m, n, d = 10,000, 10, 32:**

- L2 Ratio: ~100x Faster
- Cosine Ratio: ~300x Faster

## Features

1. **Two Distance Metrics:** Full support for L2 and Cosine Similarity.
2. **Floating Point Stability:** Implements epsilon offsets and `np.maximum` to prevent precision errors.
3. **Memory Efficiency:** Uses `argpartition` for $O(n)$ complexity when finding the top $k$ results.

## Project Structure

- `engine.py`: The core Engine class containing the VSE logic.
- `utils.py`: A `manual_test` (brute force) function for validation.
- `test_engine.py`: Testing suite for numerical precision.
- `benchmark.py`: Script to compare execution times.

## Usage

```python
import numpy as np
from engine import Engine

# 1. Initialize data
database = np.random.randn(1000, 128)
queries = np.random.randn(10, 128)

# 2. Setup Engine
engine = Engine('cosine')
engine.add_to_index(database)

# 3. Search for Top 5 results
results = engine.search(queries, k=5)
print(results)
```
