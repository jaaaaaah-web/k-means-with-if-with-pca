# ui_components.py

import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- Helper function for color mapping ---
def get_color_map(data, column):
    """Creates a mapping from unique values in a column to a list of colors."""
    unique_values = data[column].unique()
    colors = [
        "#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF",
        "#00FFFF", "#FFA500", "#800080", "#008000", "#800000"
    ]
    color_dict = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
    return color_dict

# --- Functions for displaying SINGLE analysis results ---

def display_single_result(result):
    """
    Displays the complete set of results for a single analysis run.
    """
    st.subheader(f"Results for: {result['name']}")
    
    # Display Metrics
    metrics = result['metrics']
    col1, col2, col3 = st.columns(3)
    col1.metric("Inertia", f"{metrics['inertia']:.2f}")
    col2.metric("Silhouette Score", f"{metrics['silhouette']:.2f}")
    col3.metric("Davies-Bouldin Index", f"{metrics['dbi']:.2f}")

    # Prepare data for mapping
    data = result['data']
    color_map = get_color_map(data, 'cluster')
    data['cluster_color'] = data['cluster'].map(color_map)

    # Display Map
    st.map(data, latitude='latitude', longitude='longitude', color='cluster_color')
    
    # Display Key Insights if region data is available
    if 'region' in data.columns:
        display_single_key_insights(data)

# --- Functions for COMPARING two analysis results ---

def display_evaluation_metrics(results):
    st.subheader("Evaluation Metrics Comparison")
    metrics_a = results['group_a']['metrics']
    metrics_b = results['group_b']['metrics']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Inertia (Lower is Better)", f"{metrics_b['inertia']:.2f}", f"{metrics_b['inertia'] - metrics_a['inertia']:.2f}")
    col2.metric("Silhouette Score (Higher is Better)", f"{metrics_b['silhouette']:.2f}", f"{metrics_b['silhouette'] - metrics_a['silhouette']:.2f}")
    col3.metric("Davies-Bouldin Index (Lower is Better)", f"{metrics_b['dbi']:.2f}", f"{metrics_b['dbi'] - metrics_a['dbi']:.2f}")
    st.caption("*(Change from Standard K-Means shown below each metric)*")

def display_cluster_maps(results):
    st.subheader("Cluster Visualization")
    data_a = results['group_a']['data']
    data_b = results['group_b']['data']

    # Create consistent color mapping based on enhanced results
    color_map = get_color_map(data_b, 'cluster')
    data_a['cluster_color'] = data_a['cluster'].map(color_map)
    data_b['cluster_color'] = data_b['cluster'].map(color_map)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Standard K-Means**")
        st.map(data_a, latitude='latitude', longitude='longitude', color='cluster_color')
    with col2:
        st.write("**Enhanced K-Means**")
        st.map(data_b, latitude='latitude', longitude='longitude', color='cluster_color')

# --- Shared visualization functions ---

def display_regional_map(results):
    if 'region' in results['group_b']['data'].columns:
        st.subheader("Geographic Distribution by Region")
        data = results['group_b']['data']
        
        # Create color mapping for regions
        region_color_map = get_color_map(data, 'region')
        data['region_color'] = data['region'].map(region_color_map)
        
        st.map(data, latitude='latitude', longitude='longitude', color='region_color')
        
        # Display a legend for the regions
        st.write("Region Legend:")
        legend_data = pd.DataFrame({
            'Region': region_color_map.keys(),
            'Color': [f'<div style="width:20px;height:20px;background-color:{color};"></div>' for color in region_color_map.values()]
        })
        st.write(legend_data.to_html(escape=False, index=False), unsafe_allow_html=True)


def display_regional_breakdown(results):
    if 'region' in results['group_b']['data'].columns:
        st.subheader("Fake News Count by Region")
        data = results['group_b']['data']
        region_counts = data['region'].value_counts()
        st.bar_chart(region_counts)

# --- Insight generation functions ---

@st.cache_data
def reverse_geocode_location(lat, lon):
    try:
        geolocator = Nominatim(user_agent="spatiotemporal_insights_app", timeout=10)
        location = geolocator.reverse((lat, lon), exactly_one=True, language='en')
        return location.raw.get('address', {})
    except Exception:
        return {}

def get_region_from_address(address):
    """Extracts a descriptive region name from the geocoded address dictionary."""
    if not address:
        return "Unknown Location"
    # Prioritize specific region keys, then province, then other keys
    for key in ['region', 'state', 'province', 'state_district', 'county']:
        if key in address:
            return address[key]
    # Fallback to a formatted string of available values
    return ", ".join(filter(None, [address.get('city'), address.get('country')]))

def display_single_key_insights(data):
    """Calculates and displays insights for a single result's data."""
    if not data.empty:
        # Most active region insight
        if 'region' in data.columns:
            most_active_region = data['region'].mode()[0]
        else:
            # Fallback to geocoding the center of the largest cluster
            largest_cluster = data['cluster'].mode()[0]
            cluster_data = data[data['cluster'] == largest_cluster]
            center_lat, center_lon = cluster_data['latitude'].mean(), cluster_data['longitude'].mean()
            address = reverse_geocode_location(center_lat, center_lon)
            most_active_region = get_region_from_address(address)
        
        # Peak activity time insight
        peak_hour = data['hour'].mode()[0]
        
        col1, col2 = st.columns(2)
        col1.metric("Most Active Region", most_active_region)
        col2.metric("Peak Activity Time", f"{peak_hour:02d}:00 - {peak_hour:02d}:59")

def display_key_insights(results):
    st.subheader("Key Insights from Enhanced Clustering")
    data_b = results['group_b']['data']
    display_single_key_insights(data_b)
