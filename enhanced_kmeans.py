from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np

class EnhancedKMeans(BaseEstimator, ClusterMixin):
    """
    A custom K-Means clustering algorithm that integrates Isolation Forest for outlier removal.

    This class acts as a single, cohesive algorithm that first identifies and removes
    outliers from the dataset before applying the standard K-Means algorithm to the
    cleaned data. Outliers are assigned a cluster label of -1.
    """
    def __init__(self, n_clusters=8, contamination=0.1, random_state=None, n_init=10):
        """
        Initializes the EnhancedKMeans algorithm.

        Parameters:
        - n_clusters (int): The number of clusters for the K-Means algorithm.
        - contamination (float): The proportion of outliers to detect in the Isolation Forest.
        - random_state (int): A seed for the random number generators for reproducibility.
        - n_init (int): Number of times the k-means algorithm will be run with different seeds.
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.n_init = n_init

        # Initialize the two core algorithms that this class will manage
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=self.n_init
        )

        # These will be populated after fitting
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        """
        Fits the model to the data. This involves running Isolation Forest and then K-Means.
        """
        # Step 1: Use Isolation Forest to detect outliers
        outlier_preds = self.iso_forest.fit_predict(X)
        
        # Identify the indices of the normal data points (inliers)
        inlier_mask = outlier_preds == 1
        X_cleaned = X[inlier_mask]

        # Ensure we have enough data points to form the required clusters
        if len(X_cleaned) < self.n_clusters:
            raise ValueError(f"Not enough data points ({len(X_cleaned)}) remained after outlier removal to form {self.n_clusters} clusters. Try a lower outlier percentage.")

        # Step 2: Fit the K-Means algorithm ONLY on the cleaned data
        self.kmeans.fit(X_cleaned)

        # Store the results from the fitted K-Means model
        self.cluster_centers_ = self.kmeans.cluster_centers_
        self.inertia_ = self.kmeans.inertia_

        # Step 3: Construct the final labels array for the original data
        # Create an array of -1s (outlier label) with the same length as the original data
        final_labels = np.full(X.shape[0], -1, dtype=int)
        
        # Place the cluster labels from K-Means into the correct positions for the inliers
        final_labels[inlier_mask] = self.kmeans.labels_
        
        self.labels_ = final_labels

        return self

    def fit_predict(self, X, y=None):
        """
        Fits the model and returns the cluster labels for the original data.
        """
        self.fit(X)
        return self.labels_
