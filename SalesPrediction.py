import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('my_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Function to make predictions
def predict_sales(tv_budget):
    input_data = pd.DataFrame({'TV': [tv_budget]})
    prediction = model.predict(input_data)
    return prediction[0]


# Streamlit app layout
def main():
    st.markdown(
        """
        <style>
        body {
            background-color: #F5E7B2;
        }
        .title {
            color: #E76F51;
            font-size: 3em;
            font-weight: bold;
            text-align: center;
        }
        .input-form {
            background-color: #FFFFFF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
        }
        .input-label {
            color: #36BA98;
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        .predict-button {
            background-color: #E9C46A;
            color: white;
            border: none;
            padding: 10px 20px;
            font-size: 1em;
            border-radius: 5px;
            cursor: pointer;
            text-align: center;
        }
        .predict-button:hover {
            background-color: #F4A261;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title">TV Advertising Sales Prediction</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-form">', unsafe_allow_html=True)

    # Input form
    st.markdown('<div class="input-label">TV Advertising Budget ($)</div>', unsafe_allow_html=True)
    tv_budget = st.number_input('', min_value=0.0, step=0.01)

    # Close input form div
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict button
    if st.button('Predict', key='predict-button'):
        result = predict_sales(tv_budget)
        st.write(f'Predicted Sales: ${result:.2f}')


if __name__ == '__main__':
    main()
