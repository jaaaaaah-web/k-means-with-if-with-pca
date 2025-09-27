import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator # New import for finding the elbow

# --- OBJECTIVE 3: Implement a Framework for processing spatiotemporal datasets
def prepare_data_for_clustering(df):
    """
    Extracts features from timestamp and scales the data.
    This is the essential preparation step for both models.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Feature Extraction from timestamp
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    
    # Define the features to be used for clustering
    features = ['latitude', 'longitude', 'hour', 'day_of_week']
    X = df_copy[features]

    # Scale features for distance-based algorithms like K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_copy # Return the scaled data and the df with new time features

# --- NEW FUNCTION FOR ELBOW METHOD ---
def find_optimal_k(scaled_data, k_range=(2, 11)):
    """
    Runs K-Means for a range of k and finds the optimal k using the Elbow Method.
    """
    inertias = []
    ks = range(k_range[0], k_range[1])
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Use KneeLocator to find the "elbow" of the inertia curve
    try:
        kn = KneeLocator(list(ks), inertias, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow else 4 # Default to 4 if elbow isn't found
    except Exception:
        optimal_k = 4 # Default to 4 in case of any error
        
    return inertias, optimal_k

# --- OBJECTIVE 2 & 4: Compare performance before and after applying Isolation Forest
def run_standard_analysis(df, n_clusters):
    """
    Runs the standard K-Means analysis without outlier removal.
    """
    X_scaled, df_with_features = prepare_data_for_clustering(df)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    results = {
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'dbi': davies_bouldin_score(X_scaled, labels)
        },
        'data': df_with_features.assign(cluster=labels)
    }
    return results

# --- OBJECTIVE 1: Develop an enhanced model by integrating Isolation Forest
def run_enhanced_analysis(df, n_clusters, contamination):
    """
    Runs the enhanced K-Means analysis with Isolation Forest outlier removal.
    """
    X_scaled, df_with_features = prepare_data_for_clustering(df)

    # Apply Isolation Forest to detect outliers
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outlier_preds = iso_forest.fit_predict(X_scaled)

    # Check if enough data points remain after outlier removal
    if (outlier_preds == 1).sum() < n_clusters:
        return {'error': f"Not enough data points ({ (outlier_preds == 1).sum() }) remained after outlier removal to form {n_clusters} clusters. Try a lower outlier percentage."}
        
    # Create new DataFrames without the outliers
    df_cleaned = df_with_features[outlier_preds == 1].copy()
    X_scaled_cleaned = X_scaled[outlier_preds == 1]
    
    # Run K-Means on the cleaned data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled_cleaned)

    results = {
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_scaled_cleaned, labels),
            'dbi': davies_bouldin_score(X_scaled_cleaned, labels)
        },
        'data': df_cleaned.assign(cluster=labels)
    }
    return results

