import streamlit as st
import pandas as pd
import numpy as np
st.title('Food Pickups in Bremen')
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element on the app that shows data is loading
data_load_state = st.text('Loading data, please wait...')
# Load 10,000 rows
data = load_data(10000)
# Show notification data has been loaded
data_load_state.text('Loading data...done using st.cache!')
