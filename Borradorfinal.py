import streamlit as st
import pandas as pd
from openpyxl import load_workbook

# Función para cargar el archivo Excel y asignar los datos a las variables
def cargar_datos_excel(file_path):
    try:
        # Cargar el archivo Excel
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        # Si el archivo no existe, devolvemos un DataFrame vacío
        st.error("El archivo de Excel no fue encontrado.")
        return pd.DataFrame()

# Función para guardar los datos modificados en el archivo Excel
def guardar_datos_excel(file_path, df):
    df.to_excel(file_path, index=False)
    st.success("¡Datos guardados correctamente en el archivo Excel!")

# Título de la app
st.title("Formulario de Ingreso de Datos")

# Opción para elegir el tipo de ingreso de datos
opcion = st.selectbox("Selecciona una opción para el ingreso de datos", ["Ingreso manual", "Cargar desde Excel"])

# Variable para almacenar el DataFrame de los datos
df = pd.DataFrame()

# Si se selecciona la opción de "Cargar desde Excel"
if opcion == "Cargar desde Excel":
    # Cargar el archivo Excel
    file_path = "datos_formulario.xlsx"
    df = cargar_datos_excel(file_path)

    # Verifica si hay datos en el archivo
    if not df.empty:
        # Si hay datos en el archivo, usar la primera fila para el formulario
        row = df.iloc[0]
        
        # Campos del formulario
        st.subheader("Datos Demográficos")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Edad", min_value=0, value=row['Edad'] if 'Edad' in row else 30)
        with col2:
            weight = st.number_input("Peso (kg)", min_value=0.0, value=row['Peso'] if 'Peso' in row else 70.0)
        with col3:
            length = st.number_input("Altura (cm)", min_value=0, value=row['Altura'] if 'Altura' in row else 170)

        sex = st.selectbox("Sexo", options=["Masculino", "Femenino"], index=["Masculino", "Femenino"].index(row['Sexo']) if 'Sexo' in row else 0)

        # Sección 2: Información Médica
        st.subheader("Información Médica")
        diabetes = st.selectbox("¿Tienes Diabetes?", options=["Sí", "No"], index=["Sí", "No"].index(row['Diabetes']) if 'Diabetes' in row else 0)
        hypertension = st.selectbox("¿Tienes Hipertensión?", options=["Sí", "No"], index=["Sí", "No"].index(row['Hipertensión']) if 'Hipertensión' in row else 0)
        smoker = st.selectbox("¿Eres fumador?", options=["Sí", "No"], index=["Sí", "No"].index(row['Fumador']) if 'Fumador' in row else 0)

        # Sección 3: Síntomas y Examen Físico
        st.subheader("Síntomas y Examen Físico")
        chest_pain = st.selectbox("¿Dolor en el pecho?", options=["Sí", "No"], index=["Sí", "No"].index(row['Dolor en el pecho']) if 'Dolor en el pecho' in row else 0)
        shortness_breath = st.selectbox("¿Dificultad para respirar?", options=["Sí", "No"], index=["Sí", "No"].index(row['Dificultad para respirar']) if 'Dificultad para respirar' in row else 0)

        # Sección 4: Resultados de Laboratorio
        st.subheader("Resultados de Laboratorio")
        glucose = st.number_input("Glucosa en ayunas (mg/dL)", min_value=0.0, value=row['Glucosa en ayunas'] if 'Glucosa en ayunas' in row else 0.0)
        creatinine = st.number_input("Creatinina (mg/dL)", min_value=0.0, value=row['Creatinina'] if 'Creatinina' in row else 0.0)
        lipids = st.number_input("Lípidos totales (mg/dL)", min_value=0.0, value=row['Lípidos totales'] if 'Lípidos totales' in row else 0.0)

        # Sección 5: Resultados ECG y Ecocardiografía
        st.subheader("Resultados ECG y Ecocardiografía")
        ejection_fraction = st.number_input("Fracción de eyección (%)", min_value=0.0, value=row['Fracción de eyección'] if 'Fracción de eyección' in row else 0.0)
        ecg_abnormalities = st.selectbox("¿Anomalías en ECG?", options=["Sí", "No"], index=["Sí", "No"].index(row['Anomalías ECG']) if 'Anomalías ECG' in row else 0)

        # Botón para guardar los datos modificados
        if st.button("Guardar cambios en Excel"):
            # Actualizar el DataFrame con los nuevos valores
            df.loc[0, 'Edad'] = age
            df.loc[0, 'Peso'] = weight
            df.loc[0, 'Altura'] = length
            df.loc[0, 'Sexo'] = sex
            df.loc[0, 'Diabetes'] = diabetes
            df.loc[0, 'Hipertensión'] = hypertension
            df.loc[0, 'Fumador'] = smoker
            df.loc[0, 'Dolor en el pecho'] = chest_pain
            df.loc[0, 'Dificultad para respirar'] = shortness_breath
            df.loc[0, 'Glucosa en ayunas'] = glucose
            df.loc[0, 'Creatinina'] = creatinine
            df.loc[0, 'Lípidos totales'] = lipids
            df.loc[0, 'Fracción de eyección'] = ejection_fraction
            df.loc[0, 'Anomalías ECG'] = ecg_abnormalities

            # Guardar el archivo de Excel con los datos actualizados
            guardar_datos_excel(file_path, df)

else:
    # Si se selecciona la opción de "Ingreso manual"
    st.subheader("Ingreso de Datos Manual")
    
    # Campos para ingresar datos manualmente
    age = st.number_input("Edad", min_value=0, value=30)
    weight = st.number_input("Peso (kg)", min_value=0.0, value=70.0)
    length = st.number_input("Altura (cm)", min_value=0, value=170)
    sex = st.selectbox("Sexo", options=["Masculino", "Femenino"])

    # Sección 2: Información Médica
    diabetes = st.selectbox("¿Tienes Diabetes?", options=["Sí", "No"])
    hypertension = st.selectbox("¿Tienes Hipertensión?", options=["Sí", "No"])
    smoker = st.selectbox("¿Eres fumador?", options=["Sí", "No"])

    # Sección 3: Síntomas y Examen Físico
    chest_pain = st.selectbox("¿Dolor en el pecho?", options=["Sí", "No"])
    shortness_breath = st.selectbox("¿Dificultad para respirar?", options=["Sí", "No"])

    # Sección 4: Resultados de Laboratorio
    glucose = st.number_input("Glucosa en ayunas (mg/dL)", min_value=0.0, value=0.0)
    creatinine = st.number_input("Creatinina (mg/dL)", min_value=0.0, value=0.0)
    lipids = st.number_input("Lípidos totales (mg/dL)", min_value=0.0, value=0.0)

    # Sección 5: Resultados ECG y Ecocardiografía
    ejection_fraction = st.number_input("Fracción de eyección (%)", min_value=0.0, value=0.0)
    ecg_abnormalities = st.selectbox("¿Anomalías en ECG?", options=["Sí", "No"])

    # Botón para guardar los datos manualmente
    if st.button("Guardar datos manuales"):
        # Crear un DataFrame con los datos ingresados manualmente
        df_manual = pd.DataFrame([{
            'Edad': age,
            'Peso': weight,
            'Altura': length,
            'Sexo': sex,
            'Diabetes': diabetes,
            'Hipertensión': hypertension,
            'Fumador': smoker,
            'Dolor en el pecho': chest_pain,
            'Dificultad para respirar': shortness_breath,
            'Glucosa en ayunas': glucose,
            'Creatinina': creatinine,
            'Lípidos totales': lipids,
            'Fracción de eyección': ejection_fraction,
            'Anomalías ECG': ecg_abnormalities
        }])

        # Guardar los datos en un archivo Excel
        file_path_manual = "datos_formulario_manual.xlsx"
        df_manual.to_excel(file_path_manual, index=False)
        st.success("¡Datos guardados correctamente!")

