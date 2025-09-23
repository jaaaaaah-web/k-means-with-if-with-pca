# analysis.py

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import streamlit as st

def prepare_data(df):
    """
    Extracts features from timestamp and scales the data.
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    df_copy['month'] = df_copy['timestamp'].dt.month
    
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month']
    X = df_copy[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, df_copy

def run_standard_analysis(df, n_clusters=4):
    """
    Runs the analysis for standard K-Means on the original data.
    """
    _ , X_scaled, df_features = prepare_data(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    result = {
        'name': 'Standard K-Means',
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'dbi': davies_bouldin_score(X_scaled, labels)
        },
        'data': df_features.assign(cluster=labels)
    }
    return result

def run_enhanced_analysis(df, n_clusters=4, contamination=0.1):
    """
    Runs the analysis for K-Means with Isolation Forest outlier removal.
    """
    _ , X_scaled, df_features = prepare_data(df)

    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_scaled)

    # Create new DataFrames without the outliers
    df_cleaned = df_features[outlier_preds == 1].copy()
    X_scaled_cleaned = X_scaled[outlier_preds == 1]
    
    # Check if enough data points remain for clustering
    if len(df_cleaned) < n_clusters:
        st.error(f"Not enough data remained after removing outliers to form {n_clusters} clusters. Try a lower outlier percentage.")
        return None

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled_cleaned)
    
    result = {
        'name': 'Enhanced K-Means',
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled_cleaned, labels),
            'dbi': davies_bouldin_score(X_scaled_cleaned, labels)
        },
        'data': df_cleaned.assign(cluster=labels)
    }
    return result
