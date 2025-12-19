import numpy as np 

class Engine:
    def __init__(self, metric):
        self.metric = metric.strip().lower()
        self.database = None 
        # variable to store norms of candidates in our database.
        self.db_norms = None 

        # if metric is anything other than these two, raises error.
        if self.metric not in ['l2', 'cosine']:
            raise ValueError("The metric you've chosen doesn't exist. Choose l2 or cosine as your metric")


    def add_to_index(self, D):
        # to avoid modifying global database
        self.database = D.copy() 
        if self.metric == 'cosine':
            # calculates the norm along each row (candidate). The next line divides every element in the row by its norm.
            self.db_norms = np.linalg.norm(self.database, axis=1, keepdims=True)
            self.database = self.database / (self.db_norms + 1e-10)
        elif self.metric == 'l2':
            # calculates the sum of squares for every element in each row.
            self.db_norms = np.sum(self.database**2, axis=1, keepdims=True)

    
    def search(self, Q, k):
        if self.metric == 'cosine':
            Q_normalized = np.linalg.norm(Q, axis=1, keepdims=True)
            Q = Q / (Q_normalized + 1e-10)
            product = np.matmul(Q, self.database.T)
            # using -product instead of product to get highest cosine (most similar)
            scores = -product
            

        elif self.metric == 'l2':
            Q_normalized = np.sum(Q**2, axis=1, keepdims=True)
            product = np.matmul(Q, self.database.T)
            distances = Q_normalized + self.db_norms.T - 2 * product
            # to make sure we avoid floating point errors I used np.maximum
            scores = np.maximum(distances, 0)

        top_indexes_unsorted = np.argpartition(scores, k, axis=1)[:, :k]

        # extract M rows from the array we've created
        rows = np.arange(top_indexes_unsorted.shape[0]).reshape(1,-1)
        # we create a new array of top scores, by choosing only rows and columns that are the top k choice for the respective row.
        top_scores = scores[rows, top_indexes_unsorted]
        sorted_indexes = np.argsort(top_scores, axis = 1)

        top_candidates = top_indexes_unsorted[rows, sorted_indexes]
        
        # returns a database sized (Mxk), where M is number of queries in Q, and k is the number we specificed.
        return top_candidates
