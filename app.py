# app.py

import streamlit as st
from data_processing import load_and_clean_data, filter_for_fake_news, geocode_dataframe
from analysis import run_standard_analysis, run_enhanced_analysis
from ui_components import (
    display_evaluation_metrics, display_cluster_maps, 
    display_key_insights, display_regional_breakdown, 
    display_regional_map, display_single_result
)

# --- Page Configuration ---
st.set_page_config(
    page_title="K-Means Clustering Simulation",
    page_icon="🗺️",
    layout="wide"
)

# --- App State Management ---
if 'step' not in st.session_state:
    st.session_state.step = "upload"
# Use separate dataframes for each analysis path to ensure separation
if 'df_for_standard' not in st.session_state:
    st.session_state.df_for_standard = None
if 'df_for_enhanced' not in st.session_state:
    st.session_state.df_for_enhanced = None
if 'df_pre_geocoding' not in st.session_state:
    st.session_state.df_pre_geocoding = None
# To store results
if 'standard_result' not in st.session_state:
    st.session_state.standard_result = None
if 'enhanced_result' not in st.session_state:
    st.session_state.enhanced_result = None

# --- Main App UI ---
st.title("K-Means Clustering Simulation for Spatiotemporal Data")


# --- STEP 1: UPLOAD & PREPROCESS ---
if st.session_state.step == "upload":
    st.header("Step 1: Upload & Preprocess Data")
    uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'])

    if uploaded_file:
        raw_df = load_and_clean_data(uploaded_file)
        st.dataframe(raw_df.head())
        
        st.subheader("Optional: Filter for Fake News")
        enable_filter = st.checkbox("Enable filtering for fake news")
        
        if enable_filter:
            columns = raw_df.columns.tolist()
            label_col = st.selectbox("Select the column containing credibility labels:", columns)
            filter_text = st.text_input("Enter the text that identifies fake news:", "not credible")
            
            if st.button("Apply Filter and Proceed"):
                filtered_df = filter_for_fake_news(raw_df, label_col, filter_text)
                if filtered_df is not None:
                    st.session_state.df_pre_geocoding = filtered_df
                    st.session_state.step = "mapping"
                    st.rerun()
        else:
            if st.button("Proceed without Filtering"):
                st.session_state.df_pre_geocoding = raw_df
                st.session_state.step = "mapping"
                st.rerun()
   
    

# --- STEP 2: COLUMN MAPPING & GEOCODING ---
if st.session_state.step == "mapping":
    st.header("Step 2: Map Data Columns")
    st.dataframe(st.session_state.df_pre_geocoding.head())
    columns = st.session_state.df_pre_geocoding.columns.tolist()
    loc_col = st.selectbox("Select Location column:", columns)
    time_col = st.selectbox("Select Timestamp column:", columns)
    
    has_region_col = st.checkbox("My data already has a Region column")
    region_col = st.selectbox("Select Region column:", columns) if has_region_col else None

    if st.button("Confirm Mapping & Geocode"):
        with st.spinner("Geocoding data... This may take a moment."):
            df_geocoded = geocode_dataframe(st.session_state.df_pre_geocoding, loc_col, time_col, region_col)
        
        if df_geocoded is not None:
            # Create two separate copies of the data to ensure analysis paths are distinct
            st.session_state.df_for_standard = df_geocoded.copy()
            st.session_state.df_for_enhanced = df_geocoded.copy()
            st.session_state.step = "analysis"
            st.rerun()

# --- STEP 3: SEQUENTIAL ANALYSIS & COMPARISON ---
if st.session_state.step == "analysis":
    st.sidebar.title("Analysis Parameters")
    
    # --- Standard K-Means Analysis ---
    with st.container():
        st.header("Step 3: Standard K-Means Analysis")
        st.write("This analysis uses the prepared data without any outlier removal.")
        st.dataframe(st.session_state.df_for_standard.head(3))

        st.sidebar.subheader("Standard K-Means")
        k_standard = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_std")
        
        if st.button("🚀 Run Standard Analysis"):
            with st.spinner("Running Standard K-Means..."):
                result = run_standard_analysis(st.session_state.df_for_standard, k_standard)
            if result:
                st.session_state.standard_result = result
                st.success("Standard K-Means analysis complete.")
    
    # If standard is done, show its results and the option for enhanced
    if st.session_state.standard_result:
        st.write("---")
        with st.expander("View Standard K-Means Results", expanded=False):
            display_single_result(st.session_state.standard_result)
        st.write("---")

        # --- Enhanced K-Means Analysis ---
        with st.container():
            st.header("Step 4: Enhanced K-Means Analysis")
            st.write("This analysis first cleans the data by removing outliers using Isolation Forest.")
            
            st.sidebar.subheader("Enhanced K-Means")
            k_enhanced = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_enh")
            percentage_options = [1, 5, 10, 15, 20, 25]
            selected_percentage = st.sidebar.selectbox(
                "Outlier Percentage (%)", options=percentage_options, index=2, key="contam"
            )
            contamination = selected_percentage / 100.0

            if st.button("🚀 Run Enhanced Analysis"):
                with st.spinner("Running Enhanced K-Means..."):
                    result = run_enhanced_analysis(st.session_state.df_for_enhanced, k_enhanced, contamination)
                if result:
                    st.session_state.enhanced_result = result
                    st.success("Enhanced K-Means analysis complete.")

    # If both analyses are complete, show the final comparison
    if st.session_state.standard_result and st.session_state.enhanced_result:
        st.write("---")
        st.header("Final Comparison")
        st.info("Both analyses are complete. Here is a side-by-side comparison of the results.")
        
        comparison_results = {
            'group_a': st.session_state.standard_result,
            'group_b': st.session_state.enhanced_result
        }
        
        display_evaluation_metrics(comparison_results)
        display_cluster_maps(comparison_results)
        display_regional_map(comparison_results)
        display_key_insights(comparison_results)
        display_regional_breakdown(comparison_results)

    st.sidebar.write("---")
    if st.sidebar.button("Start Over"):
        st.session_state.clear()
        st.rerun()

