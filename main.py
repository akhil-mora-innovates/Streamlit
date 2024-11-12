import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from faker import Faker

# Generate Random Lead Data
fake = Faker()

# Random Lead Data Generation
def generate_lead_data(n=200):
    data = []
    for _ in range(n):
        lead_name = fake.name()
        lead_source = np.random.choice(['Conference', 'Website Visit', 'Reference Contact'])
        conversion_probability = np.random.uniform(0, 1)  # Continuous value between 0 and 1
        data.append([lead_name, lead_source, conversion_probability])
    return pd.DataFrame(data, columns=['Lead Name', 'Lead Source', 'Conversion Probability'])

# Generate the leads dataset
df = generate_lead_data()

# Encode categorical data (Lead Source)
label_encoder = LabelEncoder()
df['Lead Source Code'] = label_encoder.fit_transform(df['Lead Source'])

# Prepare the features (X) and target (y)
X = df[['Lead Source Code']]  # Features
y = df['Conversion Probability']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
df['Predicted Conversion Probability'] = model.predict(X)

# Streamlit UI Setup
st.title("Lead Conversion Probability Prediction")

# Show charts and tables
if st.button("Show Training Data"):
    st.write("Training Data", df[['Lead Name', 'Lead Source', 'Conversion Probability']])

# Show Conversion Probability Distribution Chart
fig = px.histogram(df, x="Conversion Probability", nbins=20, title="Conversion Probability Distribution")
st.plotly_chart(fig, use_container_width=True)

# Show Lead Source Distribution Chart
fig2 = px.pie(df, names="Lead Source", title="Lead Source Distribution")
st.plotly_chart(fig2, use_container_width=True)

# Lead Details Selector (Checkboxes for selection)
st.subheader("Lead Details")

# Create checkboxes for each lead
selected_leads = []
for lead in df['Lead Name']:
    if st.checkbox(f"Select {lead}", key=lead):
        selected_leads.append(lead)

# Show Details Button
if selected_leads:
    show_details = st.button("Show Selected Lead Details")
    if show_details:
        # Display the details of the selected leads
        st.write("Showing details for the selected leads:")
        details = df[df['Lead Name'].isin(selected_leads)]
        st.write(details[['Lead Name', 'Lead Source', 'Conversion Probability', 'Predicted Conversion Probability']])
else:
    st.write("Please select leads to view details.")

# Footer Information
st.markdown("""
    **Conversion Prediction Model**
    - Uses a proprietory training model to predict the conversion probability for leads based on the lead source.
""")
