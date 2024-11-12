import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.express as px

# Sample data creation
data = {
    "Lead Name": ["Lead A", "Lead B", "Lead C", "Lead D", "Lead E", "Lead F", "Lead G", "Lead H", "Lead I", "Lead J"],
    "Recent Interactions": np.random.randint(1, 20, 10),
    "Last Engagement Days": np.random.randint(0, 30, 10),
    "Lead Source Score": np.random.randint(1, 4, 10),
    "Company Size Score": np.random.randint(1, 10, 10),
    "Converted": np.random.randint(0, 2, 10)
}

lead_data = pd.DataFrame(data)
source_mapping = {1: "Conferences", 2: "Website Visit", 3: "Reference Contact"}
lead_data["Lead Source"] = lead_data["Lead Source Score"].map(source_mapping)

# Define features and target
X = lead_data[["Recent Interactions", "Last Engagement Days", "Lead Source Score", "Company Size Score"]]
y = lead_data["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
lead_data["Conversion Probability"] = model.predict_proba(X)[:, 1]

# Streamlit UI Layout
st.title("Enhanced Predictive Lead Scoring App")
st.write("Model Accuracy:", accuracy)

# Interactive Buttons
if st.button("Refresh Data"):
    st.experimental_rerun()  # This button will refresh the app, re-running all code.

# Sidebar Selection for Lead Details
st.sidebar.header("Lead Details")
selected_lead = st.sidebar.selectbox("Select a Lead", lead_data["Lead Name"].unique())
selected_lead_data = lead_data[lead_data["Lead Name"] == selected_lead]
st.sidebar.write(f"Conversion Probability for {selected_lead}: {selected_lead_data['Conversion Probability'].values[0]:.2f}")

# Display Training Data
st.subheader("Training Data Sample")
if st.button("Show Training Data"):
    st.write("Training Data Sample:")
    st.table(X_train)

# Column Layouts for Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversion Probability by Lead")
    fig = px.bar(
        lead_data,
        x="Lead Name",
        y="Conversion Probability",
        color="Lead Source",
        title="Lead Conversion Probabilities",
        labels={"Conversion Probability": "Probability"},
    )
    st.plotly_chart(fig)

with col2:
    st.subheader("Lead Source Distribution")
    fig2 = px.pie(
        lead_data,
        names="Lead Source",
        title="Lead Source Distribution",
    )
    st.plotly_chart(fig2)

# Additional Button for Prediction
if st.button("Recalculate Probabilities"):
    lead_data["Conversion Probability"] = model.predict_proba(X)[:, 1]
    st.success("Conversion probabilities updated!")

st.write("### Detailed Lead Information")
st.table(lead_data[["Lead Name", "Lead Source", "Conversion Probability"]])
