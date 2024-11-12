import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from faker import Faker

# Generate random data
fake = Faker()
np.random.seed(42)

# Create a list of lead sources
lead_sources = ['Conference', 'Website Visit', 'Reference Contact']

# Generate 200 random lead entries
data = []
for i in range(200):
    lead_name = fake.name()
    lead_source = np.random.choice(lead_sources)
    conversion_probability = np.random.uniform(0, 1)
    data.append([lead_name, lead_source, conversion_probability])

# Create a DataFrame
df = pd.DataFrame(data, columns=["Lead Name", "Lead Source", "Conversion Probability"])

# Encode lead source as a categorical variable
label_encoder = LabelEncoder()
df['Lead Source Code'] = label_encoder.fit_transform(df['Lead Source'])

# Split data into training and testing datasets
X = df[['Lead Source Code']]
y = df['Conversion Probability']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
df['Predicted Conversion Probability'] = model.predict(X)

# Create columns for display
col1, col2 = st.columns([2, 2])

# Conversion Probability by Lead Chart
with col1:
    st.subheader("Conversion Probability by Lead")
    fig = px.bar(
        df,
        x="Lead Name",
        y="Conversion Probability",
        color="Lead Source",
        title="Lead Conversion Probabilities",
        labels={"Conversion Probability": "Probability"},
    )
    # Assign a unique key to avoid duplicate element IDs
    st.plotly_chart(fig, use_container_width=True, key="conversion_probability_chart")

# Lead Source Distribution Chart
with col2:
    st.subheader("Lead Source Distribution")
    fig2 = px.pie(
        df,
        names="Lead Source",
        title="Lead Source Distribution",
    )
    # Assign a unique key to avoid duplicate element IDs
    st.plotly_chart(fig2, use_container_width=True, key="lead_source_distribution_chart")

# Allow multiple lead selection
st.subheader("Select Lead(s) for Detailed Information")
lead_selection = st.multiselect(
    "Choose one or more leads:",
    df['Lead Name'].tolist(),
    default=df['Lead Name'].tolist()[:5]  # default to first 5 leads
)

# Button to show selected lead details
if st.button("Show Selected Lead Details"):
    if lead_selection:
        for lead in lead_selection:
            lead_details = df[df['Lead Name'] == lead].iloc[0]
            st.write(f"**Lead Name**: {lead_details['Lead Name']}")
            st.write(f"**Lead Source**: {lead_details['Lead Source']}")
            st.write(f"**Conversion Probability**: {lead_details['Conversion Probability']:.2f}")
            st.write(f"**Predicted Conversion Probability**: {lead_details['Predicted Conversion Probability']:.2f}")
            st.write("---")  # separator
    else:
        st.write("Please select at least one lead.")

# Option to show/hide detailed training data
if st.checkbox("Show/Hide Training Data"):
    st.subheader("Training Data")
    st.write(df[['Lead Name', 'Lead Source', 'Conversion Probability']])
