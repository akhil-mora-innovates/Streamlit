import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Create sample data with Lead Source Score mapped to descriptive terms
data = {
    "Lead Name": ["Lead A", "Lead B", "Lead C", "Lead D", "Lead E", "Lead F", "Lead G", "Lead H", "Lead I", "Lead J"],
    "Recent Interactions": np.random.randint(1, 20, 10),
    "Last Engagement Days": np.random.randint(0, 30, 10),
    "Lead Source Score": np.random.randint(1, 4, 10),  # Random score between 1 and 3
    "Company Size Score": np.random.randint(1, 10, 10),
    "Converted": np.random.randint(0, 2, 10)
}

lead_data = pd.DataFrame(data)

# Map Lead Source Score to descriptive terms
source_mapping = {1: "Conferences", 2: "Website Visit", 3: "Reference Contact"}
lead_data["Lead Source"] = lead_data["Lead Source Score"].map(source_mapping)

# Define features and target
X = lead_data[["Recent Interactions", "Last Engagement Days", "Lead Source Score", "Company Size Score"]]
y = lead_data["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create training and testing datasets for display
train_data = X_train.copy()
train_data['Converted'] = y_train.values
train_data["Lead Source"] = train_data["Lead Source Score"].map(source_mapping)

test_data = X_test.copy()
test_data['Converted'] = y_test.values
test_data["Lead Source"] = test_data["Lead Source Score"].map(source_mapping)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Predict probabilities for all leads
lead_data["Conversion Probability"] = model.predict_proba(X)[:, 1]

# Streamlit UI
st.title("Predictive Lead Scoring")
st.write("Model Accuracy:", accuracy)

# Show Training Data
st.subheader("Training Data")
st.table(train_data[["Recent Interactions", "Last Engagement Days", "Lead Source", "Company Size Score", "Converted"]])

# Show Testing Data
st.subheader("Testing Data")
st.table(test_data[["Recent Interactions", "Last Engagement Days", "Lead Source", "Company Size Score", "Converted"]])

# Show Lead Scoring
st.subheader("Lead Scoring with Conversion Probability")
st.table(lead_data[["Lead Name", "Lead Source", "Conversion Probability"]])
