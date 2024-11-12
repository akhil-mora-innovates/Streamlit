import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st

# Create sample data
data = {
    "Lead Name": ["Lead A", "Lead B", "Lead C", "Lead D", "Lead E", "Lead F", "Lead G", "Lead H", "Lead I", "Lead J"],
    "Recent Interactions": np.random.randint(1, 20, 10),
    "Last Engagement Days": np.random.randint(0, 30, 10),
    "Lead Source Score": np.random.randint(1, 5, 10),
    "Company Size Score": np.random.randint(1, 10, 10),
    "Converted": np.random.randint(0, 2, 10)
}

lead_data = pd.DataFrame(data)

# Define features and target
X = lead_data[["Recent Interactions", "Last Engagement Days", "Lead Source Score", "Company Size Score"]]
y = lead_data["Converted"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
st.write("Lead Scoring:")
st.table(lead_data[["Lead Name", "Conversion Probability"]])
