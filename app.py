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
    page_title="Enhanced K-Means Clustering",
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
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

# --- Main App UI ---
st.title("Enhanced K-Means Clustering for Spatiotemporal Data 🗺️")

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
        df_geocoded = geocode_dataframe(st.session_state.df_pre_geocoding, loc_col, time_col, region_col)
        
        if df_geocoded is not None:
            # Create two separate copies of the data to ensure analysis paths are distinct
            st.session_state.df_for_standard = df_geocoded.copy()
            st.session_state.df_for_enhanced = df_geocoded.copy()
            st.session_state.step = "analysis"
            st.rerun()

# --- STEP 3: RUN ANALYSIS & VIEW RESULTS ---
if st.session_state.step == "analysis":
    st.sidebar.title("Analysis Navigation")
    analysis_choice = st.sidebar.radio(
        "Choose an analysis to perform:",
        ("Standard K-Means", "Enhanced K-Means", "Compare Both"),
        key="nav_choice"
    )
    
    st.header(f"Step 3: {analysis_choice} Analysis")
    st.info("The data has been prepared (geocoded, etc.) and is ready for analysis.")
    st.dataframe(st.session_state.df_for_standard.head(3))
    st.write("---")

    # --- Standard K-Means Analysis ---
    if analysis_choice == "Standard K-Means":
        st.sidebar.header("Parameters")
        k_standard = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_std")
        
        st.subheader("Run Standard K-Means Analysis")
        st.write("This analysis uses the prepared data without any outlier removal.")
        if st.button("🚀 Run Standard Analysis"):
            with st.spinner("Running Standard K-Means..."):
                result = run_standard_analysis(st.session_state.df_for_standard, k_standard)
            if result:
                st.session_state.standard_result = result
        
        if st.session_state.standard_result:
            display_single_result(st.session_state.standard_result)

    # --- Enhanced K-Means Analysis ---
    elif analysis_choice == "Enhanced K-Means":
        st.sidebar.header("Parameters")
        k_enhanced = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_enh")
        contamination = st.sidebar.slider("Outlier Percentage", 0.01, 0.5, 0.1, key="contam")

        st.subheader("Run Enhanced K-Means Analysis")
        st.write("This analysis first cleans the data by removing outliers using Isolation Forest.")
        if st.button("🚀 Run Enhanced Analysis"):
            with st.spinner("Running Enhanced K-Means..."):
                result = run_enhanced_analysis(st.session_state.df_for_enhanced, k_enhanced, contamination)
            if result:
                st.session_state.enhanced_result = result

        if st.session_state.enhanced_result:
            display_single_result(st.session_state.enhanced_result)

    # --- Compare Both Analyses ---
    elif analysis_choice == "Compare Both":
        st.sidebar.header("Parameters")
        n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_comp")
        contamination_comp = st.sidebar.slider("Outlier Percentage", 0.01, 0.5, 0.1, key="contam_comp")

        st.subheader("Run Comparison Analysis")
        st.write("This will run both analyses with the specified parameters and show a side-by-side comparison.")
        if st.button("🚀 Run Full Comparison"):
            with st.spinner("Running both analyses..."):
                results_a = run_standard_analysis(st.session_state.df_for_standard, n_clusters)
                results_b = run_enhanced_analysis(st.session_state.df_for_enhanced, n_clusters, contamination_comp)
            
            if results_a and results_b:
                st.session_state.comparison_results = {'group_a': results_a, 'group_b': results_b}
        
        if st.session_state.comparison_results:
            results = st.session_state.comparison_results
            display_evaluation_metrics(results)
            display_cluster_maps(results)
            display_regional_map(results)
            display_key_insights(results)
            display_regional_breakdown(results)
    
    st.sidebar.write("---")
    if st.sidebar.button("Start Over"):
        st.session_state.clear()
        st.rerun()

