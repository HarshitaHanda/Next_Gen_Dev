import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('titanic.csv')
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    inputs = df[['Pclass', 'Sex', 'Age', 'Fare']]
    target = df['Survived']
    return df, inputs, target

df, inputs, target = load_data()

# Encode the 'Sex' column for model training
inputs['Sex'] = inputs['Sex'].map({'male': 'Male', 'female': 'Female'})
le_Sex = LabelEncoder()
inputs['Sex'] = le_Sex.fit_transform(inputs['Sex'])

# Train model
xtrain, xtest, ytrain, ytest = train_test_split(inputs, target, train_size=0.8, random_state=42)
model = tree.DecisionTreeClassifier()
model.fit(xtrain, ytrain)
predictions = model.predict(xtest)
accuracy = accuracy_score(ytest, predictions)

# Streamlit app
st.title('Titanic Survival Prediction')
st.write("""
This app uses a Decision Tree Classifier to predict survival on the Titanic dataset.
""")

# st.header('Model Accuracy')
# st.write(f'Accuracy of the model is: {accuracy:.2f}')

st.header('Predict Survival')
pclass = st.selectbox('Pclass', sorted(inputs['Pclass'].unique()))
sex = st.selectbox('Sex', ['Male', 'Female'])
age = st.slider('Age', float(inputs['Age'].min()), float(inputs['Age'].max()), float(inputs['Age'].mean()))
fare = st.slider('Fare', float(inputs['Fare'].min()), float(inputs['Fare'].max()), float(inputs['Fare'].mean()))

# Convert sex input to numerical value
sex_encoded = le_Sex.transform([sex])[0]

input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'Fare': [fare]
})

if st.button('Predict'):
    prediction = model.predict(input_data)[0]
    result = 'Survived' if prediction == 1 else 'Did not survive'
    st.markdown(f"<h2 style='text-align: center; color: red;'>{result}</h2>", unsafe_allow_html=True)
