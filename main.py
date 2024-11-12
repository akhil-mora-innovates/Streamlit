import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import plotly.express as px
from faker import Faker  # For generating random names

# Initialize Faker to generate random names
fake = Faker()

# Function to generate more realistic sample data with random lead names
def generate_sample_data(num_samples=200):
    np.random.seed(42)  # For reproducibility
    data = {
        "Lead Name": [fake.name() for _ in range(num_samples)],  # Random names
        "Recent Interactions": np.random.randint(5, 30, num_samples),  # Realistic interaction values
        "Last Engagement Days": np.random.randint(5, 60, num_samples),  # Engagement days ranging from 5 to 60
        "Lead Source Score": np.random.randint(1, 4, num_samples),  # Same Lead Sources: 1 - Conferences, 2 - Website Visit, 3 - Reference Contact
        "Company Size Score": np.random.randint(5, 15, num_samples),  # A range of company sizes from 5 to 15
        "Converted": np.random.randint(0, 2, num_samples)  # Binary outcome for conversion
    }
    
    lead_data = pd.DataFrame(data)
    source_mapping = {1: "Conferences", 2: "Website Visit", 3: "Reference Contact"}
    lead_data["Lead Source"] = lead_data["Lead Source Score"].map(source_mapping)
    
    return lead_data

# Initialize or regenerate data based on button click
if "lead_data" not in st.session_state or st.button("Refresh Data"):
    st.session_state["lead_data"] = generate_sample_data(num_samples=200)

lead_data = st.session_state["lead_data"]

# Define features and target
X = lead_data[["Recent Interactions", "Last Engagement Days", "Lead Source Score", "Company Size Score"]]
y = lead_data["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=500)  # Increase iterations for convergence
model.fit(X_train, y_train)

# Model accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
lead_data["Conversion Probability"] = model.predict_proba(X)[:, 1]

# Streamlit UI Layout
st.title("Enhanced Predictive Lead Scoring App")
st.write("Model Accuracy:", accuracy)

# Column Layouts for Initial Charts
col1, col2 = st.columns(2)

# Conversion Probability by Lead Chart
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
    st.plotly_chart(fig, use_container_width=True)

# Lead Source Distribution Chart
with col2:
    st.subheader("Lead Source Distribution")
    fig2 = px.pie(
        lead_data,
        names="Lead Source",
        title="Lead Source Distribution",
    )
    st.plotly_chart(fig2, use_container_width=True)

# Sidebar Selection for Lead Details (Updated to allow multiple selection)
selected_leads = st.multiselect("Select Leads", lead_data["Lead Name"].unique())

# Add the button for Show/Hide Detailed Lead Information in the main UI
if "show_detailed_info" not in st.session_state:
    st.session_state["show_detailed_info"] = False

detail_toggle_label = "Hide Lead Details" if st.session_state["show_detailed_info"] else "Show Lead Details"
if st.button(detail_toggle_label):
    st.session_state["show_detailed_info"] = not st.session_state["show_detailed_info"]

# Display Detailed Lead Information for selected leads
if st.session_state["show_detailed_info"]:
    if selected_leads:
        for lead in selected_leads:
            selected_lead_data = lead_data[lead_data["Lead Name"] == lead]
            
            st.write(f"### Details for {lead}")
            
            # Display Lead's Information in a Table
            st.write("**Lead Information**")
            st.table(selected_lead_data[["Lead Name", "Lead Source", "Recent Interactions", "Last Engagement Days", "Company Size Score"]])
            
            # Display Conversion Probability
            st.write(f"**Conversion Probability**: {selected_lead_data['Conversion Probability'].values[0]:.2f}")
            
            # Display Lead Source in a Pie Chart
            st.subheader("Lead Source Breakdown")
            fig = px.pie(
                selected_lead_data,
                names="Lead Source",
                title="Lead Source Breakdown for Selected Lead",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Please select leads to view details.")

# Toggle Show/Hide Training Data
if "show_training_data" not in st.session_state:
    st.session_state["show_training_data"] = False

toggle_label = "Hide Training Data" if st.session_state["show_training_data"] else "Show Training Data"
if st.button(toggle_label):
    st.session_state["show_training_data"] = not st.session_state["show_training_data"]

# Display Training Data with Lead Names
if st.session_state["show_training_data"]:
    # Add Lead Name to Training Data for better readability
    X_train_with_names = X_train.copy()
    X_train_with_names["Lead Name"] = lead_data.loc[X_train.index, "Lead Name"]
    st.subheader("Training Data Sample")
    st.table(X_train_with_names.set_index("Lead Name"))
