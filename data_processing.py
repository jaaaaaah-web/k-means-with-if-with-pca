# data_processing.py

import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

def load_and_clean_data(uploaded_file):
    """
    Loads data from an uploaded CSV and removes empty 'Unnamed' columns.
    """
    df = pd.read_csv(uploaded_file)
    # Drop columns that start with 'Unnamed'
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df

def filter_for_fake_news(df, label_col, filter_text):
    """
    Filters the DataFrame to keep only rows matching the fake news label.
    """
    if label_col not in df.columns:
        st.warning(f"Label column '{label_col}' not found. Skipping fake news filter.")
        return df
        
    rows_before = len(df)
    # Ensure we are working with string data to avoid errors
    df_filtered = df[df[label_col].astype(str).str.contains(filter_text, case=False, na=False)].copy()
    rows_after = len(df_filtered)
    
    st.info(f"Filtered for rows where '{label_col}' contains '{filter_text}'. Kept {rows_after} out of {rows_before} rows.")

    if rows_after == 0:
        st.error("No data remained after filtering for fake news. Please check your label column and the text you provided.")
        return None
        
    return df_filtered

@st.cache_data
def geocode_dataframe(df_processed, loc_col, time_col, region_col=None):
    """
    Takes a DataFrame and geocodes the location column.
    It also handles timestamp conversion and keeps the optional region column.
    """
    # --- PREPROCESSING STEP ---
    st.info("Preprocessing data...")
    
    # 1. Select the essential columns
    columns_to_keep = [loc_col, time_col]
    if region_col:
        columns_to_keep.append(region_col)
    
    df_clean = df_processed[columns_to_keep].copy()
    
    # 2. Drop rows with empty values in the essential columns
    rows_before = len(df_clean)
    df_clean.dropna(subset=[loc_col, time_col], inplace=True)
    rows_after = len(df_clean)
    
    if rows_after < rows_before:
        st.success(f"Preprocessing complete. Removed {rows_before - rows_after} empty rows.")
    else:
        st.success("Preprocessing complete. No empty rows found.")

    # --- Geocoding Step ---
    st.info("Starting the geocoding process. This may take a while...")
    try:
        geolocator = Nominatim(user_agent="spatiotemporal_analysis_app", timeout=10)
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        unique_locations = df_clean[loc_col].unique()
        location_dict = {}
        
        progress_bar = st.progress(0, text="Geocoding locations...")
        
        for i, loc in enumerate(unique_locations):
            location_data = geocode(f"{loc}, Philippines")
            location_dict[loc] = (location_data.latitude, location_data.longitude) if location_data else (None, None)
            progress_bar.progress((i + 1) / len(unique_locations), text=f"Geocoding: {loc}")

        progress_bar.empty()

        df_clean['latitude'] = df_clean[loc_col].map(lambda loc: location_dict.get(loc, (None, None))[0])
        df_clean['longitude'] = df_clean[loc_col].map(lambda loc: location_dict.get(loc, (None, None))[1])
        
        df_clean.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        if len(df_clean) == 0:
            st.error("Geocoding failed for all locations.")
            return None
        
        df_clean['timestamp'] = pd.to_datetime(df_clean[time_col], dayfirst=True)
        
        # Final column renaming for consistency
        final_df = df_clean.rename(columns={loc_col: 'location', time_col: 'timestamp_orig'})
        if region_col:
            final_df = final_df.rename(columns={region_col: 'region'})
            
        return final_df

    except Exception as e:
        st.error(f"An error occurred during data preparation: {e}")
        return None
