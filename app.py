import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
df = pd.read_csv('iris.csv')

# Encode the target variable 'species' using LabelEncoder
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Splitting the data into training and testing sets
x = df.drop('species', axis=1)
y = df['species']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.8, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(xtrain, ytrain)

# Streamlit app
st.title('Iris Flower Prediction Web App')

# Creating columns for input sliders
col1, col2, col3, col4 = st.columns(4)

# User input fields in the columns
sepal_length = col1.slider('Sepal Length', float(df['sepal_length'].min()), float(df['sepal_length'].max()), float(df['sepal_length'].mean()), key='sepal_length')
sepal_width = col2.slider('Sepal Width', float(df['sepal_width'].min()), float(df['sepal_width'].max()), float(df['sepal_width'].mean()), key='sepal_width')
petal_length = col3.slider('Petal Length', float(df['petal_length'].min()), float(df['petal_length'].max()), float(df['petal_length'].mean()), key='petal_length')
petal_width = col4.slider('Petal Width', float(df['petal_width'].min()), float(df['petal_width'].max()), float(df['petal_width'].mean()), key='petal_width')

# Add space between sliders and prediction
st.write("\n\n")

# Predict function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    input_data = pd.DataFrame({'sepal_length': [sepal_length],
                               'sepal_width': [sepal_width],
                               'petal_length': [petal_length],
                               'petal_width': [petal_width]})
    prediction = model.predict(input_data)
    predicted_species = le.inverse_transform(prediction)[0]
    return predicted_species

# Display prediction
predicted_species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
st.markdown(f"<h2 style='font-weight: bolder;'>Predicted Species: {predicted_species}</h2>", unsafe_allow_html=True)

# Custom CSS for styling sliders
st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        background-color: blue;
    }
    .stSlider > div > div > div > div > div[role='slider'] {
        background-color: blue;
    }
    </style>
""", unsafe_allow_html=True)
