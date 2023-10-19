"""
OpenLabCluster: Active Learning Based Clustering and Classification of Animal Behaviors in Videos Based on Automatically Extracted Kinematic Body Keypoints
Copyright (c) 2022-2023 University of Washington. Developed in UW NeuroAI Lab by Jingyuan Li.
"""
import numpy as np
from sklearn.metrics import pairwise_distances

class CoreSet():
    def __init__(self, features, metric='euclidean'):
        """
        :param features: feature for each sample
        :param metric: the measure for the distance between samples
        """
        self.features = features
        self.metric = metric
        self.selected = []
        self.nSamples = self.features.shape[0]


    def init_distance(self):
        """
        :param selected_points: previous selected points, will be treated as centers
        :return:
        """
        if self.selected is not None:
            featureCenter = self.features[self.selected]
            dist = pairwise_distances(self.features, featureCenter, metric=self.metric)
            self.min_distances = np.min(dist, axis=1).reshape(-1, 1)

    def update_distance(self, selected_points):
        """
        Update distance based on newly selected samples
        :param selected_points: previous selected points, will be treated as centers
        :return:
        """
        new_selected = [x for x in selected_points
                           if x not in self.selected]
        featureCenter =self.features[new_selected]
        dist = pairwise_distances(self.features, featureCenter, metric=self.metric)
        self.min_distances = np.minimum(self.min_distances, dist)

    def select_samples(self, feature, nBudget,selected=[]):
        """
        Select samples based on the new feature set and number of budge to be selected
        :param feature: extracted hidden state ndarray
        :param nBudget: int, number of samples to be selected
        :return:
        """
        self.features = feature
        self.selected = selected
        newSelection = []
        self.init_distance()
        for i in range(nBudget):
            if self.selected is None:
                idx = np.random.choice(np.arange(self.nSamples))
            else:
                idx = np.argmax(self.min_distances)
            assert idx not in self.selected
            self.update_distance([idx])
            newSelection.append(idx)
            self.selected.append(idx)
        return newSelection
