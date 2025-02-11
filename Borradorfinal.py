import streamlit as st
import os
import joblib
import zipfile
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gzip
import pickle
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


def load_encoder():
    with open("onehot_encoder_5.pkl", "rb") as f:
            encoder = pickle.load(f)
    with open("numerical_columns_2.pkl", "rb") as f:
            numerical_columns = pickle.load(f)
    return encoder, numerical_columns

def load_model_1():
    """Cargar el modelo y sus pesos desde el archivo model_weights.pkl."""
    # nombre de la red neuronal
    filename = 'model_trained_classifier.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model1 = pickle.load(f)
    return model1
    
def load_model_2():
    filename = 'best_model.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        model2 = pickle.load(f)
    return model2

model1=load_model_1()

model2=load_model_2()

column_names = [
            "Age", "Weight", "Length", "Sex", "BMI", "DM", "HTN", "Current Smoker", 
    "EX-Smoker", "FH", "Obesity", "CRF", "CVA",
            "Airway disease", "Thyroid Disease", "CHF", "DLP", "BP", "PR", "Edema", 
    "Weak Peripheral Pulse", "Lung rales",
            "Systolic Murmur", "Diastolic Murmur", "Typical Chest Pain", "Dyspnea", 
    "Function Class", "Atypical", "Nonanginal",
            "Exertional CP", "LowTH Ang", "Q Wave", "St Elevation", "St Depression", 
    "Tinversion", "LVH", "Poor R Progression",
            "BBB", "FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na", 
    "WBC", "Lymph", "Neut", "PLT", "EF-TTE",
            "Region RWMA"
        ]
categorical_columns = {
            "Sex": ["Male", "Female"],"DM": [0,1],"HTN":[0,1],"Current Smoker": [0, 1],"EX-Smoker": [0, 1],"FH": [0, 1],"Obesity": ["Y", "N"],
            "CRF": ["Y", "N"],"CVA": ["Y", "N"],"Airway disease": ["Y", "N"],"Thyroid Disease": ["Y", "N"],"CHF": ["Y", "N"],"Edema": [0,1],
            "Weak Peripheral Pulse": ["Y","N"],"Lung rales": ["Y","N"],"Systolic Murmur": ["Y","N"],"Diastolic Murmur": ["Y","N"],"Typical Chest Pain": [0,1],
            "Dyspnea": ["Y","N"],"Function Class": [0,1,2,3],"Atypical": ["Y","N"],"Nonanginal": ["Y","N"],"LowTH Ang": ["Y","N"],"Q Wave": [0,1],
            "St Elevation": [0,1],"St Depression": [0, 1],"Tinversion": [0, 1],"LVH": ["Y", "N"],"Poor R Progression": ["Y", "N"],"BBB": ["LBBB", "N","RBBB"], 
            "Region RWMA": [0,1,2,3,4],"VHD": ["mild","Moderate","N","Severe"]
        }

heartdisease = pd.read_csv('heartdisease.csv')

X = heartdisease.iloc[:, :-1]
y = heartdisease['Cath']
X_encoded = pd.get_dummies(X, drop_first=True,dtype= int)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
df_defecto=X_test.copy()

df=X_test.copy()
df_first_row = df.iloc[0,:].to_frame().T # Estos son los valores por defecto y no deben pasar por encoder



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


#Modelo Clasico
import pandas as pd

if st.sidebar.checkbox("Utilizar arboles de decisión"):
    st.write("### Arboles de decisión")
    st.write("""
    El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.
    """)

    st.write("### Indique si desea hacer una predicción de manera manual o cargar una fila desde un archivo Excel")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Manual', 'Cargar desde Excel'], key="madelo1_metodo_prediccion")


    if selected_column == 'Cargar desde Excel':
        st.write("### Cargar archivo Excel para la predicción")
        uploaded_file = st.file_uploader("Cargar archivo Excel", type=["xlsx"])

        if uploaded_file is not None:
            # Leer el archivo Excel
            df_excel = pd.read_excel(uploaded_file)

            # Mostrar las primeras filas para ver los datos
            st.write("### Datos cargados del archivo Excel:")
            st.write(df_excel.head())

            # Pedir al usuario que seleccione la fila (podría ser por índice o número de fila)
            row_number = st.number_input("Selecciona el número de fila para la predicción", min_value=0, max_value=len(df_excel)-1, value=0)

            # Seleccionar la fila correspondiente
            selected_row = df_excel.iloc[row_number, :]

            # Mostrar la fila seleccionada
            st.write("### Fila seleccionada para la predicción:")
            st.write(selected_row)

            # Preparar los datos para la predicción: aplicar One Hot Encoder y separación de variables numéricas
            encoder, numerical_columns = load_encoder()

            # Separar variables categóricas y numéricas
            new_data_categorical = selected_row[encoder.feature_names_in_].to_frame().T  # Convertir a DataFrame
            new_data_numerical = selected_row[numerical_columns].to_frame().T  # Convertir a DataFrame

            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)

            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())

            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)

            # Realizar la predicción
            prediction = model1.predict(final_data)

            if prediction == 1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)


    if selected_column=='Manual':             
        # Crear DataFrame inicial con valores numéricos en 0 y categóricos con el primer valor de la lista
        data = {col: [0.0] for col in column_names}  # Inicializar numéricos en 0
        for col in categorical_columns:
            data[col] = [categorical_columns[col][0]]  # Inicializar con el primer valor de la lista
        
        df = pd.DataFrame(data)
        
        # Convertir columnas categóricas a tipo "category" para que se muestren como dropdown en st.data_editor
        for col in categorical_columns:
            df[col] = df[col].astype("category")
        
        # Mostrar la tabla editable en Streamlit
        st.write("### Introduce los datos para la predicción:")
        edited_df = st.data_editor(df, key="editable_table")
        
        # Mostrar la tabla actualizada
        st.write("#### Datos ingresados:")
        st.write(edited_df)
        
        # Botón para generar la predicción
        if st.button("Realizar predicción"):
            st.write("Procesando los datos para la predicción...")
            # Mostrar los datos originales
            st.write(" **Datos originales:**")
            st.write(edited_df)
            
            encoder, numerical_columns = load_encoder()
            
            # Simulación de datos nuevos
            new_data = edited_df
            # Separar variables numéricas y categóricas
            new_data_categorical = new_data[encoder.feature_names_in_]  # Mantiene solo las categóricas
            new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas
            
            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)
            
            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
            
            # Concatenar las variables numéricas con las categóricas codificadas
            final_data = pd.concat([new_data_numerical, encoded_df], axis=1)
    
            prediction=model1.predict(final_data)
            if prediction==1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)
                


from sklearn.preprocessing import StandardScaler

if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
           
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)

    st.write("### Indique si desea hacer una predicción de manera manual o cargar un archivo Excel")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Cargar archivo Excel', 'Manual'], key="madelo2_metodo_prediccion")

    if selected_column == 'Cargar archivo Excel':
        st.write("### Cargar el archivo Excel con los datos para la predicción")

        # Cargar archivo Excel
        uploaded_file = st.file_uploader("Sube un archivo Excel", type="xlsx")
        
        if uploaded_file is not None:
            # Leer archivo Excel
            df = pd.read_excel(uploaded_file)
            st.write("Datos cargados:", df.head())

            # Obtener las columnas que espera el modelo
            encoder, numerical_columns = load_encoder()
            encoded_column_names = encoder.get_feature_names_out()
            numerical_column_names = numerical_columns

            # Las columnas esperadas por el modelo
            expected_columns = list(encoded_column_names) + list(numerical_column_names)
            st.write("Las columnas esperadas por el modelo son:", expected_columns)

            # Compara las columnas cargadas con las esperadas
            loaded_columns = df.columns.tolist()
            st.write("Las columnas cargadas desde el archivo Excel son:", loaded_columns)

            missing_columns = [col for col in expected_columns if col not in loaded_columns]
            extra_columns = [col for col in loaded_columns if col not in expected_columns]

            if missing_columns:
                st.write("Faltan las siguientes columnas en los datos cargados:", missing_columns)
            if extra_columns:
                st.write("Las siguientes columnas no deberían estar presentes:", extra_columns)

            # Paso 1: Codificar las columnas categóricas
            # Asumimos que las columnas categóricas son aquellas en la lista `categorical_columns`
            categorical_columns = list(set(encoder.feature_names_in_) & set(loaded_columns))  # Filtra las columnas categóricas presentes

            # Si hay columnas categóricas, las codificamos
            if categorical_columns:
                # Codificar las variables categóricas
                categorical_data = df[categorical_columns]
                encoded_array = encoder.transform(categorical_data)
                encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
                st.write("Datos categóricos codificados:", encoded_df.head())
            else:
                st.write("No se encontraron columnas categóricas en los datos.")

            # Paso 2: Separar columnas numéricas
            numerical_columns_in_data = list(set(numerical_columns) & set(loaded_columns))  # Filtra las columnas numéricas presentes
            numerical_data = df[numerical_columns_in_data]
            
            # Paso 3: Escalar las columnas numéricas (estandarización)
            scaler = StandardScaler()
            numerical_data_scaled = scaler.fit_transform(numerical_data)
            numerical_data_scaled = pd.DataFrame(numerical_data_scaled, columns=numerical_columns_in_data)
            
            # Paso 4: Concatenar las columnas numéricas escaladas con las columnas categóricas codificadas
            final_data = pd.concat([numerical_data_scaled, encoded_df], axis=1)

            # Paso 5: Asegurarse de que el DataFrame tenga las columnas en el orden esperado
            final_data = final_data[expected_columns]  # Ajustar al orden de las columnas esperadas por el modelo
            st.write("Datos ajustados para la predicción:", final_data.head())

            # Paso 6: Realizar la predicción
            if not final_data.empty:
                prediction_probabilities = model2.predict(final_data)

                # Probabilidades de la clase 0 y 1
                prob_normal = 1 - prediction_probabilities
                prob_cath = prediction_probabilities
                
                # Mostrar las probabilidades de cada clase
                st.write(f"### Probabilidades:")
                st.write(f"Probabilidad de Normal: {prob_normal[0][0]:.4f}")
                st.write(f"Probabilidad de Cath: {prob_cath[0][0]:.4f}")
                
                # Decidir la predicción (usando un umbral de 0.5)
                prediction = 1 if prob_cath > 0.5 else 0
                
                if prediction == 1:
                    st.write("Predicción del modelo: Cath")
                else:
                    st.write("Predicción del modelo: Normal")
            else:
                st.write("No se encontraron datos válidos para hacer la predicción.")
    
    if selected_column == 'Manual':
        # Crear DataFrame inicial con valores numéricos en 0 y categóricos con el primer valor de la lista
        data = {col: [0.0] for col in column_names}  # Inicializar numéricos en 0
        for col in categorical_columns:
            data[col] = [categorical_columns[col][0]]  # Inicializar con el primer valor de la lista
        
        df = pd.DataFrame(data)
        
        # Convertir columnas categóricas a tipo "category" para que se muestren como dropdown en st.data_editor
        for col in categorical_columns:
            df[col] = df[col].astype("category")
        
        # Mostrar la tabla editable en Streamlit
        st.write("### Introduce los datos para la predicción:")
        edited_df = st.data_editor(df, key="editable_table")
        
        # Mostrar la tabla actualizada
        st.write("#### Datos ingresados:")
        st.write(edited_df)
        
        # Botón para generar la predicción
        if st.button("Realizar predicción"):
            st.write("Procesando los datos para la predicción...")
            # Mostrar los datos originales
            st.write(" **Datos originales:**")
            st.write(edited_df)
            
            encoder, numerical_columns = load_encoder()
            
            # Simulación de datos nuevos
            new_data = edited_df
            # Separar variables numéricas y categóricas
            new_data_categorical = new_data[encoder.feature_names_in_]  # Mantiene solo las categóricas
            new_data_numerical = new_data[numerical_columns]  # Mantiene solo las numéricas
            
            # Codificar las variables categóricas
            encoded_array = encoder.transform(new_data_categorical)
            
            # Convertir la salida a DataFrame con nombres de columnas codificadas
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out())
            
            # Escalar las columnas numéricas
            numerical_data_scaled = scaler.transform(new_data_numerical)  # Usamos el mismo scaler entrenado
            numerical_data_scaled = pd.DataFrame(numerical_data_scaled, columns=numerical_columns)
            
            # Concatenar las variables numéricas escaladas con las categóricas codificadas
            final_data = pd.concat([numerical_data_scaled, encoded_df], axis=1)

            # Asegurarse de que las columnas estén en el orden correcto
            final_data = final_data[expected_columns]  # Reordenar para que coincidan con las columnas esperadas
            
            # Realizar la predicción
            prediction_probabilities = model2.predict(final_data)
            
            # Probabilidades de la clase 0 y 1
            prob_normal = 1 - prediction_probabilities
            prob_cath = prediction_probabilities
            
            # Mostrar las probabilidades de cada clase
            st.write(f"### Probabilidades:")
            st.write(f"Probabilidad de Normal: {prob_normal[0][0]:.4f}")
            st.write(f"Probabilidad de Cath: {prob_cath[0][0]:.4f}")
            
            # Decidir la predicción (usando un umbral de 0.5)
            prediction = 1 if prob_cath > 0.5 else 0
            
            if prediction == 1:
                st.write("Predicción del modelo: Cath")
            else:
                st.write("Predicción del modelo: Normal")




additional_params = {
    'Depth': 1,
    'Epochs': 11,
    'Batch Size': 58,
    'Accuracy': 0.704918,
    'Loss': 0.6126
}

# Colocar el checkbox en la barra lateral
if st.sidebar.checkbox("Mostrar hiperparámetros del modelo"):
    st.write("#### Hiperparámetros del modelo")
    
    # Mostrar los hiperparámetros del modelo 1 (modelo de clasificación - árbol de decisión)
    if hasattr(model1, 'get_params'):
        st.write("##### Hiperparámetros del modelo de clasificación (sklearn)")

        model1_params = model1.get_params()  # Extraer los hiperparámetros del modelo

        # Convertir los hiperparámetros a un formato adecuado para una tabla
        model1_params_table = [(key, value) for key, value in model1_params.items()] 

        # Limpiar los valores None o <NA> y reemplazarlos con un guion o valor vacío
        cleaned_model1_params = [
            (key, value if value is not None and value != "<NA>" else "-") 
            for key, value in model1_params_table
        ]

        # Mostrar los parámetros del modelo 1 como una tabla
        model1_params_df = pd.DataFrame(cleaned_model1_params, columns=["Hiperparámetro", "Valor"])

        # Establecer el ancho de las columnas para que se ajusten adecuadamente
        model1_params_df.style.set_properties(subset=["Hiperparámetro", "Valor"], width="300px")

        # Mostrar la tabla con estilo
        st.dataframe(model1_params_df, use_container_width=True)

        # Agregar tabla con el Accuracy del modelo de árbol de decisión (con 6 decimales)
        st.write("##### Accuracy del modelo de clasificación (Árbol de Decisión)")
        accuracy_params = {
            "Accuracy": f"{0.836065:.6f}"  # Formatear el Accuracy con 6 decimales
        }
        accuracy_df = pd.DataFrame(list(accuracy_params.items()), columns=["Métrica", "Valor"])
        st.dataframe(accuracy_df, use_container_width=True)

    # Mostrar los hiperparámetros del modelo 2 (modelo de red neuronal)
    if hasattr(model2, 'get_config'):
        st.write("##### Hiperparámetros del modelo de red neuronal (TensorFlow/Keras)")

        # Obtener los hiperparámetros de la red neuronal
        model2_params = []
        for layer in model2.layers:
            layer_info = {
                "Capa": layer.__class__.__name__,  # Nombre de la capa (ej. Dense, Conv2D)
                "Hiperparámetros": layer.get_config()  # Obtiene la configuración de la capa
            }
            model2_params.append(layer_info)

        # Crear un diccionario para almacenar los hiperparámetros de cada capa
        layers_info = {}

        for i, layer in enumerate(model2_params):
            layer_name = f"Capa {i+1} ({layer['Capa']})"
            layer_config = layer["Hiperparámetros"]

            for param, value in layer_config.items():
                if param not in layers_info:
                    layers_info[param] = []
                layers_info[param].append(value)

        # Convertir el diccionario en un DataFrame con los hiperparámetros como filas
        model2_params_df = pd.DataFrame(layers_info)

        # Transponer la tabla para que las capas estén como columnas y los hiperparámetros como filas
        model2_params_df = model2_params_df.transpose()

        # Renombrar las columnas para reflejar el número de capa
        model2_params_df.columns = [f"Capa {i+1}" for i in range(len(model2_params))]

        # Establecer el ancho de las columnas para que se ajusten adecuadamente
        model2_params_df.style.set_properties(subset=model2_params_df.columns, width="300px")

        # Mostrar la tabla con estilo
        st.dataframe(model2_params_df, use_container_width=True)

        # Obtener el learning rate
        if hasattr(model2, 'optimizer'):
            optimizer = model2.optimizer
            if hasattr(optimizer, 'lr'):  # Para versiones más antiguas de Keras
                learning_rate = optimizer.lr.numpy()
            elif hasattr(optimizer, 'learning_rate'):  # Para versiones más recientes de TensorFlow
                learning_rate = optimizer.learning_rate.numpy()

        # Agregar el learning rate a los parámetros generales
        additional_params['Learning Rate'] = learning_rate

        # Crear un DataFrame para los parámetros generales
        additional_params_df = pd.DataFrame(list(additional_params.items()), columns=["Hiperparámetro", "Valor"])

        # Ajustar los decimales de los valores para que se muestren con hasta 6 decimales
        def format_value(value):
            if isinstance(value, (float, int)):
                # Si el valor tiene decimales, mostrarlo con 6 decimales, de lo contrario, mostrarlo como entero
                if value.is_integer():
                    return f"{int(value)}"  # Mostrar como entero si no tiene decimales
                return f"{value:.6f}"  # Mostrar con 6 decimales
            return value  # Para valores no numéricos, devolver tal cual

        additional_params_df["Valor"] = additional_params_df["Valor"].apply(format_value)

        # Mostrar la tabla de los parámetros generales
        st.write("##### Parámetros Generales del Modelo")
        st.dataframe(additional_params_df, use_container_width=True)

    else:
        st.write("El modelo no tiene el método get_config() disponible.")




