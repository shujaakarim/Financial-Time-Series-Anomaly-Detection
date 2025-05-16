import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Smart Data Visualizer", layout="wide")

st.title("📊 Smart Data Visualizer Dashboard")
st.markdown("""
Upload any CSV file and explore your dataset with customizable graphs and anomaly detection.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File successfully loaded!")

        # Show first few rows in a scrollable container
        with st.expander("🔍 Preview Data"):
            st.dataframe(df.head(), height=250)

        # Sidebar - Column selection
        st.sidebar.header("📌 Plot Customization")
        x_col = st.sidebar.selectbox("Select X-axis column", df.columns)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        y_col = st.sidebar.selectbox("Select Y-axis column (numeric only)", numeric_cols)

        chart_type = st.sidebar.radio("Select Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot"])

        # Plotting
        st.subheader(f"📈 {chart_type} for {y_col} vs {x_col}")
        fig, ax = plt.subplots(figsize=(10, 4))
        if chart_type == "Line Chart":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "Bar Chart":
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        elif chart_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)

        # Anomaly detection (simple)
        with st.expander("🚨 Anomaly Detection (Z-score based)"):
            z_thresh = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=3.0)
            if y_col in numeric_cols:
                df["z_score"] = (df[y_col] - df[y_col].mean()) / df[y_col].std()
                anomalies = df[np.abs(df["z_score"]) > z_thresh]
                st.write(f"Found {len(anomalies)} potential anomalies:")
                st.dataframe(anomalies[[x_col, y_col, "z_score"]])

    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
else:
    st.info("📁 Please upload a CSV file to get started.")
