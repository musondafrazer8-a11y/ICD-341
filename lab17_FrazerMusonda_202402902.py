import streamlit as st
import numpy as np
import pandas as pd

# Set the title of the dashboard
st.title("ICD 341: Interactive Statistical Dashboard")
st.write("This dashboard demonstrates how to deploy a statistical app using Streamlit.")

# Create a sidebar for user inputs
st.sidebar.header("User Inputs")

# Slider for sample size
sample_size = st.sidebar.slider("Select Sample Size:", min_value=10, max_value=1000, value=100, step=10)

# Dropdown for distribution type
distribution = st.sidebar.selectbox("Select Distribution:", ["Normal", "Uniform", "Exponential"])

# Generate data based on user selection
np.random.seed(42)
if distribution == "Normal":
    data = np.random.normal(loc=50, scale=10, size=sample_size)
elif distribution == "Uniform":
    data = np.random.uniform(low=0, high=100, size=sample_size)
else:
    data = np.random.exponential(scale=20, size=sample_size)

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Generated Values"])

# Display the first few rows of the data
st.subheader("Data Preview")
st.dataframe(df.head())

# Display descriptive statistics
st.subheader("Summary Statistics")
st.write(df.describe())

# Display a line chart
st.subheader("Line Chart of Data")
st.line_chart(df)

# Display a histogram
st.subheader("Histogram")
# Streamlit has a built-in histogram function
hist_values = np.histogram(data, bins=20, range=(data.min(), data.max()))[0]
st.bar_chart(hist_values)