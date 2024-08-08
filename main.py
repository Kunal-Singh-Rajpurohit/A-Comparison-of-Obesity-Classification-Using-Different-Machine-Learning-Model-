import streamlit as st
import pandas as pd
import pickle
import os

# Load the model from a pickle file
def load_model():
    pickle_file_path = 'best_model.pkl'
    if os.path.exists(pickle_file_path) and os.path.getsize(pickle_file_path) > 0:
        with open(pickle_file_path, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error("Model file not found or is empty.")
        return None

# Preprocess input data
def preprocess_data(data):
    features = ['Gender', 'Age', 'Height', 'Weight', 'HIST_OVERWEIGHT', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
    
    data = pd.get_dummies(data, columns=['Gender', 'HIST_OVERWEIGHT', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'])
    
    data = data.reindex(columns=features, fill_value=0)
    
    return data

# Map numeric predictions to descriptive labels
def map_predictions(prediction):
    labels = {1: 'Low Obesity Risk', 2: 'Moderate Obesity Risk', 3: 'High Obesity Risk', 4: 'Very High Obesity Risk'}
    return labels.get(prediction, 'Unknown')

# Streamlit app
def main():
    st.title("Obesity Level Estimation")

    # Load model
    model = load_model()
    if model is None:
        st.stop()

    # User inputs
    st.sidebar.header("Input Features")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    age = st.sidebar.slider("Age", 14, 35, 23)
    height = st.sidebar.slider("Height (cm)", 40, 200, 160)
    weight = st.sidebar.slider("Weight (kg)", 30, 150, 70)
    hist_overweight = st.sidebar.selectbox("Family History of Overweight", ["No", "Yes"])
    favc = st.sidebar.selectbox("Frequency of Consuming High Caloric Food", ["No", "Yes"])
    fcvc = st.sidebar.slider("Frequency of Consumption of Vegetables and Fruits", 1, 3, 2)
    ncp = st.sidebar.slider("Number of Main Courses Consumed in a Day", 1, 4, 2)
    caec = st.sidebar.selectbox("Consumption of Food Between Meals", ["No", "Sometimes", "Frequently"])
    smoke = st.sidebar.selectbox("Smoking Habit", ["No", "Yes"])
    ch2o = st.sidebar.slider("Calories Consumed from Soft Drinks", 1.0, 5.0, 2.0)
    scc = st.sidebar.selectbox("Consumption of Sweetened Foods", ["No", "Sometimes", "Frequently"])
    faf = st.sidebar.slider("Physical Activity Frequency", 0.0, 3.0, 1.0)
    tue = st.sidebar.slider("Time Spent in Physical Activity (minutes per week)", 0, 180, 60)
    calc = st.sidebar.selectbox("Consumption of Alcohol", ["No", "Yes"])
    mtrans = st.sidebar.selectbox("Transportation to Work", ["Car", "Public Transport", "Walking", "Bicycle"])

    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'HIST_OVERWEIGHT': [hist_overweight],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })

    # Preprocess input data
    input_data = preprocess_data(input_data)
    
    # Debug: Show preprocessed input data
    st.write("Preprocessed Input Data:")
    st.write(input_data)

    # Predict obesity level when button is clicked
    if st.sidebar.button('Predict'):
        if model:
            prediction = model.predict(input_data)
            # Map numeric prediction to descriptive label
            prediction_label = map_predictions(prediction[0])
            # Debug: Show the prediction
            st.write("Prediction Array:")
            st.write(prediction)
            st.write(f'Predicted Obesity Level: {prediction_label}')
        else:
            st.error("Model is not loaded properly.")

if __name__ == "__main__":
    main()