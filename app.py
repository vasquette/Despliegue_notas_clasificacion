import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Predicción de Aprobación de Curso')

# Load the trained model and preprocessing objects
try:
    onehot_encoder = joblib.load('onehot_encoder.joblib')
    minmax_scaler = joblib.load('minmax_scaler.joblib')
    best_stacking_model = joblib.load('best_stacking_model.joblib')
except FileNotFoundError:
    st.error("Error: Make sure 'onehot_encoder.joblib', 'minmax_scaler.joblib', and 'best_stacking_model.joblib' are in the same directory.")
    st.stop()

# Define the preprocessing function
def preprocess_input(examen_admision, felder):
    # Create a DataFrame from user input
    input_df = pd.DataFrame({'Examen_admisión': [examen_admision], 'Felder': [felder]})

    # One-hot encode 'Felder'
    felder_encoded = onehot_encoder.transform(input_df[['Felder']])
    felder_feature_names = onehot_encoder.get_feature_names_out(['Felder'])
    felder_encoded_df = pd.DataFrame(felder_encoded, columns=felder_feature_names)

    # Scale 'Examen_admisión'
    examen_admision_scaled = minmax_scaler.transform(input_df[['Examen_admisión']])
    examen_admision_scaled_df = pd.DataFrame(examen_admision_scaled, columns=['Examen_admisión'])

    # Combine the features
    processed_input = pd.concat([examen_admision_scaled_df, felder_encoded_df], axis=1)

    return processed_input

# Streamlit UI
st.header('Ingrese los datos del estudiante')

examen_admision_input = st.number_input('Examen de Admisión', min_value=0.0, max_value=5.0, step=0.01)
felder_options = ['activo', 'visual', 'equilibrio', 'intuitivo', 'reflexivo', 'secuencial', 'sensorial', 'verbal'] # Replace with actual Felder categories
felder_input = st.selectbox('Estilo de Aprendizaje (Felder)', felder_options)


if st.button('Predecir Aprobación'):
    # Preprocess the input
    processed_data = preprocess_input(examen_admision_input, felder_input)

    # Make prediction
    prediction = best_stacking_model.predict(processed_data)

    # Display the prediction
    st.subheader('Resultado de la Predicción:')
    if prediction[0] == 'si':
        st.success('El estudiante probablemente APROBARÁ el curso.')
    else:
        st.error('El estudiante probablemente NO APROBARÁ el curso.')

st.header('Instrucciones para ejecutar la aplicación:')
st.write("1. Guarde este código como `app.py`.")
st.write("2. Asegúrese de tener los archivos `onehot_encoder.joblib`, `minmax_scaler.joblib`, y `best_stacking_model.joblib` en el mismo directorio.")
st.write("3. Abra una terminal en ese directorio.")
st.write("4. Ejecute el comando: `streamlit run app.py`")
