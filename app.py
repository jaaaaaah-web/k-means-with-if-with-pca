import streamlit as st
import pandas as pd
from data_processing import load_and_clean_data, filter_for_fake_news, geocode_dataframe, auto_detect_columns
from analysis import run_standard_analysis, run_enhanced_analysis
from ui_components import (
    display_evaluation_metrics,
    display_spatial_visualizations,
    display_temporal_patterns,
    display_dynamic_interpretation
)

# --- App Configuration ---
st.set_page_config(
    page_title="K-Means Clustering Simulation",
    page_icon="🗺️",
    layout="wide"
)

# --- App State Management ---
if 'step' not in st.session_state:
    st.session_state.step = "upload"
if 'data' not in st.session_state:
    st.session_state.data = None
if 'detected_cols' not in st.session_state:
    st.session_state.detected_cols = {}
if 'data_for_standard' not in st.session_state:
    st.session_state.data_for_standard = None
if 'data_for_enhanced' not in st.session_state:
    st.session_state.data_for_enhanced = None
if 'standard_results' not in st.session_state:
    st.session_state.standard_results = None
if 'enhanced_results' not in st.session_state:
    st.session_state.enhanced_results = None
if 'region_col' not in st.session_state:
    st.session_state.region_col = None

# --- Main App UI ---
st.title("K-Means Clustering Simulation")

# --- Introduction Text (only on the first step) ---
if st.session_state.step == "upload":
    st.markdown("This simulation framework is designed to evaluate an enhanced K-Means clustering model for analyzing spatiotemporal data.")
    st.subheader("Project Objectives:")
    st.markdown("""
    - **Develop an Enhanced Model:** Integrate Isolation Forest for outlier detection with K-Means clustering.
    - **Compare Performance:** Evaluate K-Means before and after applying Isolation Forest using key metrics.
    - **Implement a Framework:** Design this interactive simulation for processing spatiotemporal datasets.
    - **Validate Effectiveness:** Assess the model's ability to produce interpretable clusters.
    """)

st.write("---")

# --- STEP 1: UPLOAD & FILTER ---
if st.session_state.step == "upload":
    st.header("Step 1: Upload & Preprocess Data")
    uploaded_file = st.file_uploader("Upload your CSV data", type=['csv'])

    if uploaded_file:
        raw_data = load_and_clean_data(uploaded_file)
        if not raw_data.empty:
            st.session_state.data = raw_data
            st.dataframe(st.session_state.data.head())

            detected = auto_detect_columns(st.session_state.data.columns)
            
            if st.checkbox("Optional: Filter for Fake News"):
                label_col_index = list(st.session_state.data.columns).index(detected['label']) if detected['label'] else 0
                label_col = st.selectbox("Select the credibility label column:", st.session_state.data.columns, index=label_col_index)
                
                filter_text = st.text_input("Enter text that identifies fake news (e.g., 'not credible'):", "not credible")
                
                if st.button("Apply Filter and Proceed to Mapping"):
                    filtered_data = filter_for_fake_news(st.session_state.data, label_col, filter_text)
                    if filtered_data is not None and not filtered_data.empty:
                        st.session_state.data = filtered_data
                        st.success("Data filtered successfully!")
                        st.session_state.step = "mapping"
                        st.rerun()
            else:
                if st.button("Proceed to Mapping"):
                    st.session_state.step = "mapping"
                    st.rerun()

# --- STEP 2: COLUMN MAPPING (AUTOMATED WITH OVERRIDE) ---
if st.session_state.step == "mapping":
    st.header("Step 2: Confirm Data Columns")
    st.info("The system has automatically detected the columns for analysis. Please review and confirm, or correct them if needed.")

    all_columns = st.session_state.data.columns.tolist()
    if not st.session_state.detected_cols:
         st.session_state.detected_cols = auto_detect_columns(all_columns)
    
    loc_col = st.selectbox("Location Column:", all_columns, index=all_columns.index(st.session_state.detected_cols['location']) if st.session_state.detected_cols['location'] else 0)
    time_col = st.selectbox("Timestamp Column:", all_columns, index=all_columns.index(st.session_state.detected_cols['timestamp']) if st.session_state.detected_cols['timestamp'] else 1)
    region_col = st.selectbox("Region Column (Optional):", [None] + all_columns, index=all_columns.index(st.session_state.detected_cols['region']) + 1 if st.session_state.detected_cols['region'] else 0)

    if st.button("Confirm Columns & Prepare Data"):
        with st.spinner("Preparing data... This may take a while."):
            geocoded_data = geocode_dataframe(st.session_state.data, loc_col, time_col, region_col)

        if geocoded_data is not None and not geocoded_data.empty:
            st.session_state.data_for_standard = geocoded_data.copy()
            st.session_state.data_for_enhanced = geocoded_data.copy()
            st.session_state.region_col = 'region' if region_col else None
            st.session_state.step = "analysis"
            st.rerun()

# --- STEP 3: RUN ANALYSIS ---
if st.session_state.step == "analysis":
    st.header("Step 3: Run Analysis")
    st.sidebar.header("Analysis Parameters")
    
    # CONSOLIDATED PARAMETERS IN SIDEBAR
    st.sidebar.write("Set the parameters for the analyses:")
    n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, key="k_clusters")
    
    st.sidebar.write("---") 
    st.sidebar.write("Parameter for Enhanced Analysis only:")
    contamination = st.sidebar.selectbox(
        "Estimated Outlier Percentage",
        [1, 5, 10, 15, 20, 25],
        index=2 
    ) / 100.0

    st.info("First, run the baseline Standard K-Means analysis. Then, run the Enhanced K-Means analysis to see the improvement.")
    st.write("---")

    # Part A: Standard K-Means
    st.subheader("Standard K-Means Analysis")

    if st.button("Run Standard Analysis"):
        with st.spinner("Running Standard K-Means..."):
            st.session_state.standard_results = run_standard_analysis(st.session_state.data_for_standard, n_clusters)
        st.rerun()

    # Part B: Enhanced K-Means (appears after standard is done)
    if st.session_state.standard_results:
        st.subheader("Standard K-Means Results Preview")
        
        metrics = st.session_state.standard_results['metrics']
        col1, col2, col3 = st.columns(3)
        col1.metric("Inertia", f"{metrics['inertia']:.2f}")
        col2.metric("Silhouette Score", f"{metrics['silhouette']:.2f}")
        col3.metric("Davies-Bouldin Index", f"{metrics['dbi']:.2f}")
        
        display_spatial_visualizations(st.session_state.standard_results, None, single_view=True, title="Standard K-Means Scatter Plot")
        
        st.write("---")

        st.subheader("Enhanced K-Means Analysis")

        if st.button("Run Enhanced Analysis"):
            with st.spinner("Running Enhanced K-Means..."):
                st.session_state.enhanced_results = run_enhanced_analysis(st.session_state.data_for_enhanced, n_clusters, contamination)
            st.rerun()
            
        # --- Preview for Enhanced K-Means Results ---
        if st.session_state.enhanced_results:
            st.subheader("Enhanced K-Means Results Preview")
            if 'error' in st.session_state.enhanced_results:
                st.error(f"Analysis failed: {st.session_state.enhanced_results['error']}")
            else:
                metrics_enh = st.session_state.enhanced_results['metrics']
                col1_enh, col2_enh, col3_enh = st.columns(3)
                col1_enh.metric("Inertia", f"{metrics_enh['inertia']:.2f}")
                col2_enh.metric("Silhouette Score", f"{metrics_enh['silhouette']:.2f}")
                col3_enh.metric("Davies-Bouldin Index", f"{metrics_enh['dbi']:.2f}")
                
                display_spatial_visualizations(None, st.session_state.enhanced_results, single_view=True, title="Enhanced K-Means Scatter Plot")


    # Part C: Final Comparison (appears after both are done)
    if st.session_state.standard_results and st.session_state.enhanced_results:
        st.write("---")
        st.header("Final Comparison and Results")

        if 'error' in st.session_state.enhanced_results:
             pass
        else:
            # Display the core results first
            display_evaluation_metrics(st.session_state.standard_results, st.session_state.enhanced_results)
            display_spatial_visualizations(st.session_state.standard_results, st.session_state.enhanced_results)
            display_temporal_patterns(st.session_state.enhanced_results)

            # --- CALL THE DYNAMIC INTERPRETATION FUNCTION ---
            display_dynamic_interpretation(st.session_state.standard_results, st.session_state.enhanced_results)

