import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # NEW: Import PCA
from kneed import KneeLocator

# --- NEW: Import your custom EnhancedKMeans algorithm ---
from enhanced_kmeans import EnhancedKMeans

def prepare_data_for_clustering(df, n_components=None):
    """
    Extracts features from timestamp, scales the data, and applies PCA.
    """
    df_copy = df.copy()
    df_copy['hour'] = df_copy['timestamp'].dt.hour
    df_copy['day_of_week'] = df_copy['timestamp'].dt.dayofweek
    
    features = ['latitude', 'longitude', 'hour', 'day_of_week']
    X = df_copy[features]

    # Step 1: Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Apply PCA if a number of components is specified
    if n_components is not None:
        pca = PCA(n_components=n_components, random_state=42)
        X_processed = pca.fit_transform(X_scaled)
    else:
        X_processed = X_scaled
        
    return X_processed, df_copy

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
    
    try:
        kn = KneeLocator(list(ks), inertias, curve='convex', direction='decreasing')
        optimal_k = kn.elbow if kn.elbow else 4
    except Exception:
        optimal_k = 4
        
    return inertias, optimal_k

def run_standard_analysis(df, n_clusters, n_components=None):
    """
    Runs the standard K-Means analysis with optional PCA.
    """
    X_processed, df_with_features = prepare_data_for_clustering(df, n_components)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_processed)
    
    results = {
        'metrics': {
            'inertia': kmeans.inertia_,
            'silhouette': silhouette_score(X_processed, labels),
            'dbi': davies_bouldin_score(X_processed, labels)
        },
        'data': df_with_features.assign(cluster=labels)
    }
    return results

def run_enhanced_analysis(df, n_clusters, contamination, n_components=None):
    """
    Runs the enhanced analysis using the new custom EnhancedKMeans algorithm
    with optional PCA.
    """
    X_processed, df_with_features = prepare_data_for_clustering(df, n_components)

    try:
        # --- USE YOUR NEW ALGORITHM ---
        enhanced_model = EnhancedKMeans(
            n_clusters=n_clusters,
            contamination=contamination,
            random_state=42
        )
        labels = enhanced_model.fit_predict(X_processed)
        # --- END OF NEW IMPLEMENTATION ---

        # Filter out the outliers (labeled as -1) for metric calculation
        inlier_mask = labels != -1
        X_inliers = X_processed[inlier_mask]
        labels_inliers = labels[inlier_mask]
        
        if len(labels_inliers) < 2:
             return {'error': "Not enough data points remained after outlier removal to calculate performance metrics."}

        results = {
            'metrics': {
                'inertia': enhanced_model.inertia_,
                'silhouette': silhouette_score(X_inliers, labels_inliers),
                'dbi': davies_bouldin_score(X_inliers, labels_inliers)
            },
            'data': df_with_features.assign(cluster=labels)
        }
        return results

    except ValueError as e:
        return {'error': str(e)}