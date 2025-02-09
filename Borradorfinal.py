import streamlit as st
import os
import joblib
import zipfile
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from PIL import Image
import gzip
import pickle


heartdisease = pd.read_csv('heartdisease.csv')

# Título de la aplicación
st.title("Exploración de datos: Heart Disease")
st.image('heartdisease.jpg', caption="Problemas del corazón")
# Descripción inicial
st.markdown("""
Las enfermedades cardiovasculares son muy comunes y representan una de las principales causas de muerte. 
Entre los principales tipos de estas enfermedades, el diagnóstico correcto y a tiempo de la enfermedad arterial coronaria (CAD) es de suma importancia. 

Aunque la angiografía es el método más preciso para diagnosticar CAD, presenta muchos efectos secundarios y es costoso. 
Estudios previos han utilizado varias características al recopilar datos de pacientes, aplicando diferentes algoritmos de minería de datos para lograr métodos con alta precisión, menos efectos secundarios y menores costos.

En este contexto, se presenta un conjunto de datos llamado **Z-Alizadeh Sani**, que contiene información de 303 pacientes y 54 características, utilizando varios atributos efectivos para el análisis.
""")
st.write("""
### ¡Bienvenidos!
Esta aplicación interactiva permite explorar el dataset de Heart Disease.
Puedes:
1. Ver los primeros registros.
2. Consultar información general del dataset.
3. Mirar las estadisticas descriptivas.
4. Identificar los datos faltantes.
5. Analizar la frecuencia de las columnas.
6. Observar la información de cada paciente.
7. Explorar la matriz de correlación.
8. Generar graficos dinámicos.

Y además, transformar los datos mediante la imputación de datos faltantes, la codificación de variables categóricas y la estandarización de los datos.
""")

# Sección para explorar el dataset
st.sidebar.header("Exploración de datos")

def load_model():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""

    # nombre de la red neuronal
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model1 = pickle.load(f)

    filename = 'best_model.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model2 = pickle.load(f)
    return model1, model2


# Mostrar informacion en el dataset
if st.sidebar.checkbox("Conoce un poco sobre la base de datos"):
    
    st.title("Descripción de la base de datos **Z-Alizadeh Sani**")
    
    st.markdown("""
    La base de datos **Z-Alizadeh Sani** contiene información médica de pacientes cardíacos y está estructurada en cuatro categorías principales:
    """)
    
    # Demográficos
    st.subheader("1. Demográficos")
    st.markdown("""
    - **Edad:** 30-86 años.
    - **Peso:** 48-120 kg.
    - **Altura:** 140-188 cm.
    - **Sexo:** M (Masculino), F (Femenino).
    - **Índice de Masa Corporal (BMI):** 18-41.
    - **Historial médico:** Diabetes Mellitus (DM), Hipertensión (HTN), fumador actual/ex-fumador, historial familiar (FH), obesidad, insuficiencia renal crónica (CRF), enfermedad pulmonar, etc.
    """)
    
    # Síntomas y Examen Físico
    st.subheader("2. Síntomas y Examen Físico")
    st.markdown("""
    - **Presión Arterial (BP):** 90-190 mmHg.
    - **Pulso (PR):** 50-110 ppm.
    - **Síntomas cardíacos:** Edema, dolor torácico, disnea, soplos sistólicos y diastólicos.
    - **Clase funcional:** Valores del 1 al 4.
    """)
    
    # ECG
    st.subheader("3. ECG")
    st.markdown("""
    - **Anomalías observadas:** Elevación del ST, inversión de T, hipertrofia ventricular izquierda (LVH), etc.
    """)
    
    # Laboratorio y Ecocardiografía
    st.subheader("4. Laboratorio y Ecocardiografía")
    st.markdown("""
    - **Pruebas de laboratorio:**
      - Glucosa en ayunas (FBS): 62-400 mg/dl.
      - Creatinina (Cr): 0.5-2.2 mg/dl.
      - Lípidos (LDL, HDL, TG).
      - Otros valores como hemoglobina, potasio, sodio, leucocitos, etc.
    - **Ecocardiografía:**
      - Fracción de eyección (EF-TTE): 15-60%.
      - Anormalidades del movimiento regional de la pared (RWMA).
    """)


# Mostrar las primeras filas dinámicamente
if st.sidebar.checkbox("Mostrar primeras filas"):
    n_rows = st.sidebar.slider("Número de filas a mostrar:", 1, 50, 5)
    st.write(f"### Primeras {n_rows} filas del dataset")
    st.write(heartdisease.head(n_rows))


# Mostrar información del dataset
import io
if st.sidebar.checkbox("Mostrar información del dataset"):
    st.write("### Información del dataset")

    # Capturar la salida de info() en un buffer
    buffer = io.StringIO()
    heartdisease.info(buf=buffer)
    
    # Procesar la salida para estructurarla mejor
    info_text = buffer.getvalue().split("\n")  # Dividir en líneas
    info_text = [line.strip() for line in info_text if line.strip()]  # Quitar espacios vacíos
    
    # Extraer información clave
    filas_columnas = info_text[0]  # Primera línea con shape
    columnas_info = info_text[3:]  # A partir de la cuarta línea están las columnas

    # Mostrar filas y columnas
    st.write(f"**{filas_columnas}**")

    # Convertir la información de columnas en un DataFrame
    column_data = []
    for line in columnas_info:
        parts = line.split()  # Separar por espacios
        if len(parts) >= 3:
            column_name = parts[1]  # Nombre de la columna
            non_null_count = parts[2]  # Cantidad de valores no nulos
            dtype = parts[-1]  # Tipo de dato
            column_data.append([column_name, non_null_count, dtype])

    df_info = pd.DataFrame(column_data, columns=["Columna", "No Nulos", "Tipo de Dato"]).iloc[2:]
    memory_values = df_info.iloc[-1].values
    memorie_use = " ".join(str(value) for value in memory_values)
    # Mostrar la tabla en Streamlit
    st.dataframe(df_info.iloc[:-2])
    st.write(f"Uso en memoria {memorie_use}")

# Estadísticas descriptivas
if st.sidebar.checkbox("Mostrar estadísticas descriptivas"):
    st.write("### Estadísticas descriptivas")
    st.write(heartdisease.describe())
    
# Datos faltantes
if st.sidebar.checkbox("Mostrar datos faltantes"):
    st.write("### Datos faltantes por columna")
    selected_column = st.selectbox("Selecciona una columna para ver los datos faltantes:", heartdisease.columns)

    # Calcular datos faltantes
    missing_values = heartdisease[selected_column].isnull().sum()
    total_values = len(heartdisease[selected_column])
    missing_percentage = (missing_values / total_values) * 100

    # Mostrar resultado
    st.write(f"### Información de la columna: `{selected_column}`")
    st.write(f"- **Valores totales:** {total_values}")
    st.write(f"- **Valores faltantes:** {missing_values} ({missing_percentage:.2f}%)")
    
    if st.button("Mostrar todos los valores faltantes"):
        missing_total = heartdisease.isnull().sum()
        missing_total_df = pd.DataFrame({"Columna": missing_total.index, "Valores Faltantes": missing_total.values})
        
        # Filtrar solo las columnas con valores faltantes
        missing_total_df = missing_total_df[missing_total_df["Valores Faltantes"] > 0]
        st.write(missing_total_df)

#Frecuencia Columnas
if st.sidebar.checkbox("Frecuencia columnas"):
    st.write("### Frecuencia por columna")
    columna_seleccionada = st.selectbox("Selecciona una columna para ver su frecuencia:", heartdisease.columns)
    st.write(heartdisease[columna_seleccionada].value_counts())
    if st.button("Mostrar valor más frecuente"):
        st.write(heartdisease[columna_seleccionada].mode()[0])

#Informacion por paciente
if st.sidebar.checkbox("Información paciente"):
    st.write("### Informacion por paciente")
    row_index = st.number_input("Ingresa el índice de la fila a visualizar:", min_value=0, max_value=len(heartdisease)-1, step=1)

    if st.button("Mostrar fila seleccionada"):
        st.write(f"### Datos de la fila `{row_index}`")
        st.dataframe(heartdisease.iloc[[row_index]].iloc[:, 1:])

#Matriz de correlacion
if st.sidebar.checkbox("Matriz de correlacion"):
    st.write("### Matriz de correlacion")
    # Filtrar solo las columnas numéricas
    heartdisease_num = heartdisease.select_dtypes(include=['float64', 'int64'])
    variables_objetivo = ['Age','Weight','Length', 'BMI', 'BP','PR','FBS','CR','TG','LDL','HDL','BUN','ESR','HB','K','Na','WBC','Lymph','Neut','PLT','EF-TTE']
    # Calcular la matriz de correlación
    correlacion = heartdisease_num[variables_objetivo].corr()
    # Create a mask using numpy's triu function
    mask = np.triu(np.ones_like(correlacion, dtype=bool))
    # Configuración de la gráfica
    # Create a masked heatmap
    plt.figure(figsize = (10,8))
    plt.rcParams.update({'font.size': 12})
    sns.heatmap(correlacion, cmap = 'coolwarm', annot_kws={"size": 7},vmin = -1, vmax = 1, center = 0, annot=True, fmt=".2f", square=True, linewidths=.5, mask = mask)
    plt.show()

    #plt.figure(figsize=(10, 8))  # Tamaño de la figura
    #sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt='.2f', cbar=True, square=True)
    
    # Título de la gráfica
    plt.title('Matriz de Correlación de Heart Disease')
    
    # Mostrar la gráfica en Streamlit
    st.pyplot(plt)

# Sección para gráficos dinámicos
if st.sidebar.checkbox("Gráficos dinámicos"):

    # Selección de variables para el gráfico
    x_var = st.sidebar.selectbox("Selecciona la variable X:", heartdisease.columns)
    y_var = st.sidebar.selectbox("Selecciona la variable Y:", heartdisease.columns)
    
    # Tipo de gráfico
    chart_type = st.sidebar.radio(
        "Selecciona el tipo de gráfico:",
        ("Dispersión", "Histograma", "Boxplot")
    )
    
    # Mostrar el gráfico
    st.write("### Gráficos")
    if chart_type == "Dispersión":
        st.write(f"#### Gráfico de dispersión: {x_var} vs {y_var}")
        fig, ax = plt.subplots()
        sns.scatterplot(data=heartdisease, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Histograma":
        st.write(f"#### Histograma de {x_var}")
        fig, ax = plt.subplots()
        sns.histplot(heartdisease[x_var], bins=30, kde=True, ax=ax)
        st.pyplot(fig)
    elif chart_type == "Boxplot":
        st.write(f"#### Boxplot de {y_var} por {x_var}")
        fig, ax = plt.subplots()
        sns.boxplot(data=heartdisease, x=x_var, y=y_var, ax=ax)
        st.pyplot(fig)

st.sidebar.header("Transformacion datos")
# Copiar el DataFrame para evitar modificar el original
if 'heartdisease_copy' not in st.session_state:
    st.session_state.heartdisease_copy = heartdisease.copy()

if st.sidebar.checkbox("Datos categoricos"):
    # Estrategias de codificación disponibles
    estrategias2 = ['Ordinal Encoder', 'OneHot Encoder']
    
    # Crear un selectbox para seleccionar la estrategia de codificación
    strategy2 = st.selectbox('Selecciona una estrategia de codificación:', estrategias2, index=0)
    
    # Función para aplicar la codificación
    def apply_encoding(data, strategy):
        categorical_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
        st.write(f'{categorical_cols}')
        st.write(f'{len(categorical_cols)}')
        if len(categorical_cols) == 0:
            st.warning("No hay columnas categóricas en los datos.")
            return data
    
        data_copy = data.copy()
    
        if strategy2 == 'Ordinal Encoder':
            encoder = OrdinalEncoder()
            data_copy[categorical_cols] = encoder.fit_transform(data_copy[categorical_cols])
        elif strategy2 == 'OneHot Encoder':
            encoder = OneHotEncoder(sparse_output=False)
            encoded_data = pd.DataFrame(encoder.fit_transform(data_copy[categorical_cols]),
                                        columns=encoder.get_feature_names_out(categorical_cols),
                                        index=data_copy.index)
            data_copy = data_copy.drop(categorical_cols, axis=1)
            data_copy = pd.concat([data_copy, encoded_data], axis=1)
    
        return data_copy
    
    # Botón para aplicar la estrategia de codificación
    if st.button('Aplicar Estrategia de Codificación'):
        encoded_data = apply_encoding(heartdisease, strategy2)
        
        # Mostrar los datos codificados
        st.write(f"Vista previa de los datos codificados usando '{strategy2}':")
        st.dataframe(encoded_data.head())
        st.write(f"Información de los datos codificados:")
        st.write(encoded_data.info())
        st.session_state.heartdisease_copy = encoded_data.copy()

if st.sidebar.checkbox("Escalado de datos"):
    st.write("### Escalado de datos")
    # Estrategias disponibles
    estrategias1 = ['Standard Scaler', 'MinMax Scaler', 'Robust Scaler']

    # Crear selectbox para seleccionar estrategia
    strategy = st.selectbox('Selecciona una estrategia de escalado:', estrategias1, index=0)
    
    # Función para aplicar el escalado
    def apply_scaling(data, strategy):
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    
        if len(numeric_cols) == 0:
            st.warning("No hay columnas numéricas en los datos.")
            return data
    
        data_copy = data.copy()
    
        if strategy == 'Standard Scaler':
            scaler = StandardScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
        elif strategy == 'MinMax Scaler':
            scaler = MinMaxScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
        elif strategy == 'Robust Scaler':
            scaler = RobustScaler()
            data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
    
        return data_copy
    
    # Botón para aplicar la estrategia
    if st.button('Aplicar Estrategia de Escalado'):
        
        scaled_data = apply_scaling(st.session_state.heartdisease_copy, strategy)
        
        # Mostrar los datos escalados
        st.write(f"Vista previa de los datos escalados usando '{strategy}':")
        st.dataframe(scaled_data.head())
        
# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
    
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)


    
    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'])
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        model_path = None
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.endswith(".h5"):
                    model_path = os.path.join(root, file)
                    break
                    
        if model_path:
            # Cargar el modelo
            model = tf.keras.models.load_model(model_path)
            #st.success("Modelo cargado correctamente.")
            X = heartdisease.iloc[:, :-1]
            y = heartdisease['Cath']
            X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_encoded)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

            st.write("### Indique los datos por defecto que desea uasr para la predicción")
            data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'])

            if data_model=='Datos 1':
                input_data = X_train[0].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 2':
                input_data = X_train[1].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 3':
                input_data = X_train[2].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 4':
                input_data = X_train[3].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            if data_model=='Datos 5':
                input_data = X_train[4].reshape(1, -1)  # Excluir la última columna si es la etiqueta
                st.write("Datos de entrada:", input_data)

            
            # Realizar predicción
            prediction = model.predict(input_data)
            st.write("Predicción del modelo:", prediction)
        else:
            st.error("No se encontró un archivo .h5 en el ZIP. Verifica el contenido.")


 if hasattr(model, 'get_params'):
                    model_params = model.get_params()

                    # Convertir los hiperparámetros a un formato adecuado para una tabla
                    model_params_table = [(key, value) for key, value in model_params.items()]

                    # Reemplazar <NA> o None por un guion o valor vacío
                    cleaned_model_params = [
                        (key, value if value is not None and value != "<NA>" else "-") 
                        for key, value in model_params_table
                    ]

                    # Convertir a un DataFrame de pandas para tener control sobre la tabla
                    st.session_state.model_params = pd.DataFrame(cleaned_model_params, columns=["Hiperparámetro", "Valor"])

    # Checkbox para mostrar los hiperparámetros
    if st.checkbox("Mostrar hiperparámetros del modelo"):
        st.write("#### Hiperparámetros del modelo")
        if st.session_state.model_params is not None:
            # Estilo HTML para controlar el ancho de las columnas
            st.markdown(
                """
                <style>
                .dataframe th, .dataframe td {
                    padding: 10px;
                    text-align: left;
                    width: 300px;
                }
                </style>
                """, unsafe_allow_html=True
            )

            # Mostrar la tabla con estilo CSS para un ancho adecuado
            st.write(st.session_state.model_params.to_html(index=False, escape=False), unsafe_allow_html=True)
            
    if selected_column=='Manual':
        st.write("")
        

    
