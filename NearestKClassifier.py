# Author: Peter Gonzalez
import numpy as np
from collections import Counter
import random


class NearestKClassifier():
    def __init__(self, k=3):
        self.k = k

    def initializeDataSets(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidian_distance(self, coordinate1, coordinate2):
        # Euclidian formula
        return np.sqrt(np.sum((coordinate1-coordinate2)**2))

    def predict(self, X):
        predicted_labels = [self.predictHelper(x) for x in X]
        return np.array(predicted_labels)

    def predictHelper(self, x):
        # Compute the distance
        distances = [euclidian_distance(x, x_train)
                     for x_train in self.X_train]

        # Get k nearest samples and labels
        k_neighbors = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_neighbors]

        # Identify most common label
        most_common = Counter(k_nearest_labels).most_common(2)

        # Tie break if neccessary
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            if random.random() < 0.5:
                return most_common[0][0]
            else:
                return most_common[1][0]
        return most_common[0][0]
