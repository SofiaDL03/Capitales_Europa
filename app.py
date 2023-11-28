import streamlit as st
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
import pickle


# Cargar el modelo guardado
#model = tf.keras.models.load_model("titanic_model_nuevo.h5", encoding="latin1")

model = pickle.load(open('modelo_svr.pkl', 'rb'))

# Crear un objeto StandardScaler para escalar los datos de entrada
scaler = StandardScaler()

# Crear la aplicación Streamlit
def main():
    st.title("¿Cuál es mi esperanza de vida?")

    st.write("Consulta el índice de aqluiler, el índice de precios de supermercados y el índice de precios de restaurantes en tu ciudad aquí: https://es.numbeo.com/coste-de-vida/iniciar-p%C3%A1gina")
    st.write("Consulta el índice del poder adquisitivo y el índice de sanidad en tu ciudad aquí: https://es.numbeo.com/calidad-de-vida/iniciar-p%C3%A1gina")
    
    # Crear entradas para las características del pasajero
    ind_alquiler = st.number_input("Índice de alquiler de tu ciudad", min_value=0, max_value=200, value=25)
    ind_comestibles = st.number_input("Índice precios de los supermercados", min_value=0, max_value=200, value=25)
    ind_rest = st.number_input("Índice de precios de restaurantes", min_value=0, max_value=200, value=25)
    ind_adquisicion = st.number_input("Índice poder adquisitivo local", min_value=0, max_value=200, value=25)
    ind_sanidad = st.number_input("Índice de sanidad", min_value=0, max_value=200, value=25)
    
    # Crear una matriz NumPy con los datos de entrada
    input_data = np.array([[ind_alquiler, ind_comestibles, ind_rest, ind_adquisicion, ind_sanidad]])
    
    # Escalar los datos de entrada
    input_data = scaler.fit_transform(input_data)
    
    # Realizar la predicción usando el modelo cargado
    if st.button("Predecir esperanza de vida"):
        prediction = model.predict(input_data)
        probability = prediction[0]
        
        st.write(f"Esperanza de vida: {probability:.2f}")

if __name__ == "__main__":
    main()
