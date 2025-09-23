import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def _prepare_data_for_clustering(df):
    """Internal helper to extract features and scale the data."""
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    features = ['latitude', 'longitude', 'hour', 'day_of_week', 'month']
    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df

def run_standard_analysis(df_raw, n_clusters):
    """Runs K-Means on the raw, prepared data."""
    st.info("Preparing data for standard analysis...")
    X_scaled, df_features = _prepare_data_for_clustering(df_raw.copy())
    
    st.info(f"Running Standard K-Means with K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    results = {
        'name': 'Standard K-Means',
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'dbi': davies_bouldin_score(X_scaled, labels)
        },
        'data': df_features.assign(cluster=labels)
    }
    return results

def run_enhanced_analysis(df_raw, n_clusters, contamination):
    """Runs Isolation Forest to remove outliers, then K-Means."""
    st.info("Preparing data for enhanced analysis...")
    X_scaled, df_features = _prepare_data_for_clustering(df_raw.copy())

    st.info(f"Detecting outliers with Isolation Forest (contamination={contamination:.0%})...")
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_scaled)

    df_cleaned = df_features[outlier_preds == 1].copy()
    X_scaled_cleaned = X_scaled[outlier_preds == 1]
    
    st.success(f"Removed {len(df_features) - len(df_cleaned)} outliers. {len(df_cleaned)} data points remaining.")

    # --- ENHANCEMENT: Check if enough data points are left ---
    if len(df_cleaned) < n_clusters:
        st.error(f"Stopping analysis. After removing outliers, only {len(df_cleaned)} data points remain, which is not enough to form {n_clusters} clusters. Please try a lower outlier percentage or use a larger dataset.")
        return None
    
    st.info(f"Running Enhanced K-Means with K={n_clusters} on cleaned data...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled_cleaned)
    
    results = {
        'name': 'Enhanced K-Means',
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled_cleaned, labels),
            'dbi': davies_bouldin_score(X_scaled_cleaned, labels)
        },
        'data': df_cleaned.assign(cluster=labels)
    }
    return results
