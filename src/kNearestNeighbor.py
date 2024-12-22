import pandas as pd
import numpy as np
import pickle
import concurrent.futures
from os import cpu_count
from tqdm import tqdm
import time

class KNN:
    def __init__(self, k=5, n_jobs=1, metric='minkowski', p=2, weights='uniform'):
        if k < 1 or not isinstance(k, int):
            raise ValueError("Invalid k. k must be an integer greater than 0.")
        if metric not in ['manhattan', 'euclidean', 'minkowski']:
            raise ValueError("Invalid metric. Valid metrics are 'manhattan', 'euclidean', and 'minkowski'.")
        if p < 1 or not isinstance(p, (int, float)):
            raise ValueError("Invalid p. p must be a number greater than 0.")
        if weights not in ['uniform', 'distance']:
            raise ValueError("Invalid weights. Choose 'uniform' or 'distance'.")
        if n_jobs < 1 and n_jobs != -1 or not isinstance(n_jobs, int):
            raise ValueError("Invalid n_jobs. Must be an integer greater than 0, or -1 to use all available cores.")
        
        self.k = k
        self.metric = metric
        self.p = p if metric == 'minkowski' else (1 if metric == 'manhattan' else 2)
        self.weights = weights
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def get_params(self, deep=True):
        return {
            "k": self.k,
            "metric": self.metric,
            "p": self.p,
            "weights": self.weights,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _compute_distances(self, test):

        distances = np.linalg.norm(self.X_train - test, ord=self.p, axis=1)
        return distances

    def fit(self, X_train, y_train):
    
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values.astype(float)
        else:
            self.X_train = np.array(X_train).astype(float)
        self.y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train

    def predict(self, X_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values.astype(float)
        else:
            X_test = np.array(X_test, dtype=float)
        
        def predict_instance(row):
            distances = self._compute_distances(row)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train.iloc[nearest_indices]

            if self.weights == 'distance':
                nearest_distances = distances[nearest_indices]
                weights = 1 / (nearest_distances + 1e-10)  # Hindari pembagian dengan nol
                weights /= np.sum(weights)
                weighted_votes = {}
                for label, weight in zip(nearest_labels, weights):
                    if label == 0:
                        weight *= 1.5
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                prediction = max(weighted_votes, key=weighted_votes.get)
            else:
                prediction = nearest_labels.value_counts().idxmax()  # Uniform voting
            return prediction

        start_time = time.time()
        if self.n_jobs != 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                predictions = list(tqdm(executor.map(predict_instance, X_test), total=len(X_test)))
        else:
            predictions = [predict_instance(row) for row in tqdm(X_test)]
        elapsed_time = time.time() - start_time
        print(f"Prediction completed in {elapsed_time:.2f} seconds.")

        return np.array(predictions)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            return pickle.load(file)