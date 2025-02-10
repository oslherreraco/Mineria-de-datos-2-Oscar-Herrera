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
if st.sidebar.checkbox("Utilizar arboles de decisión"):
    st.write("### Arboles de decisión")
    st.write("""
    El modelo utilizado consiste en un arbol con una profundidad de 3.
    La base de datos fue codificada con One Hot Encoder y los datos no fueron escalados.
    """)
    
    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'],key="madelo1_metodo_prediccion")
    
    if selected_column=='Por defecto':
        # Buscar el archivo del modelo dentro de la carpeta extraída
        st.write("### Indique los datos por defecto que desea uasr para la predicción")
        data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'],key="modelo1_eleccion_datos")

        if data_model=='Datos 1':
            n=0
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló1")

        if data_model=='Datos 2':
            n=1
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló2")
        if data_model=='Datos 3':
            n=2
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló3")
        if data_model=='Datos 4':
            n=3
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló4")
        if data_model=='Datos 5':
            n=4
            prediction = model1.predict(df.iloc[n,:].to_frame().T)
            if prediction==1 and y_test[n]==1:
                st.write("Predicción del modelo:","Cath", prediction)
                st.write("Clasificación real","Cath", y_test[n])
                st.write("El modelo acertó")                    
            if prediction==0 and y_test[n]==0:
                st.write("Predicción del modelo:","Normal", prediction)
                st.write("Clasificación real","Normal", y_test[n])
                st.write("El modelo acertó")
            else:
                st.write("Predicción del modelo:", prediction)
                st.write("Clasificación real", y_test[n])
                st.write("El modelo falló5")

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




# Modelo de redes neuronales
if st.sidebar.checkbox("Utilizar redes Neuronales"): 
    st.write("### Redes Neuronales")
    st.write("### dwd Neuronales")
    
    
    st.write("""
    El modelo utilizado consiste en una red neuronal de una capa con 32 neuronas de entrada.
    La base de datos fue codificada con One Hot Encoder y estandarizada con StandardScaler.
    """)

    st.write("### Indique si desea hacer una predicción de manera manual o usar datos por defecto")
    selected_column = st.selectbox("Selecciona un método para la predicción", ['Por defecto','Manual'],key="madelo2_metodo_prediccion")
    if selected_column=='Por defecto':
            st.write("### Indique los datos por defecto que desea uasr para la predicción")
            data_model = st.selectbox("Selecciona un método para la predicción", ['Datos 1','Datos 2','Datos 3','Datos 4','Datos 5'],key="modelo2_eleccion_datos")

            if data_model=='Datos 1':
                n=0
                prediction = np.argmax(model2.predict(df.iloc[n,:].to_frame().T))
                if prediction==1 and y_test[n]==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                    st.write("Clasificación real","Cath", y_test[n])
                    st.write("El modelo acertó")                    
                if prediction==0 and y_test[n]==0:
                    st.write("Predicción del modelo:","Normal", prediction)
                    st.write("Clasificación real","Normal", y_test[n])
                    st.write("El modelo acertó")
                else:
                    st.write("Predicción del modelo:", prediction)
                    st.write("Clasificación real", y_test[n])
                    st.write("El modelo falló")
    
            if data_model=='Datos 2':
                n=1
                prediction = np.argmax(model2.predict(df.iloc[n,:].to_frame().T))
                if prediction==1 and y_test[n]==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                    st.write("Clasificación real","Cath", y_test[n])
                    st.write("El modelo acertó")                    
                if prediction==0 and y_test[n]==0:
                    st.write("Predicción del modelo:","Normal", prediction)
                    st.write("Clasificación real","Normal", y_test[n])
                    st.write("El modelo acertó")
                else:
                    st.write("Predicción del modelo:", prediction)
                    st.write("Clasificación real", y_test[n])
                    st.write("El modelo falló")
            if data_model=='Datos 3':
                n=2
                prediction = np.argmax(model2.predict(df.iloc[n,:].to_frame().T))
                if prediction==1 and y_test[n]==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                    st.write("Clasificación real","Cath", y_test[n])
                    st.write("El modelo acertó")                    
                if prediction==0 and y_test[n]==0:
                    st.write("Predicción del modelo:","Normal", prediction)
                    st.write("Clasificación real","Normal", y_test[n])
                    st.write("El modelo acertó")
                else:
                    st.write("Predicción del modelo:", prediction)
                    st.write("Clasificación real", y_test[n])
                    st.write("El modelo falló")
            if data_model=='Datos 4':
                n=3
                prediction = np.argmax(model2.predict(df.iloc[n,:].to_frame().T))
                if prediction==1 and y_test[n]==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                    st.write("Clasificación real","Cath", y_test[n])
                    st.write("El modelo acertó")                    
                if prediction==0 and y_test[n]==0:
                    st.write("Predicción del modelo:","Normal", prediction)
                    st.write("Clasificación real","Normal", y_test[n])
                    st.write("El modelo acertó")
                else:
                    st.write("Predicción del modelo:", prediction)
                    st.write("Clasificación real", y_test[n])
                    st.write("El modelo falló")
            if data_model=='Datos 5':
                n=4
                prediction = np.argmax(model2.predict(df.iloc[n,:].to_frame().T))
                if prediction==1 and y_test[n]==1:
                    st.write("Predicción del modelo:","Cath", prediction)
                    st.write("Clasificación real","Cath", y_test[n])
                    st.write("El modelo acertó")                    
                if prediction==0 and y_test[n]==0:
                    st.write("Predicción del modelo:","Normal", prediction)
                    st.write("Clasificación real","Normal", y_test[n])
                    st.write("El modelo acertó")
                else:
                    st.write("Predicción del modelo:", prediction)
                    st.write("Clasificación real", y_test[n])
                    st.write("El modelo falló")
    
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
    
            prediction=model2.predict(final_data)
            if prediction==1:
                st.write("Predicción del modelo:","Cath", prediction)
            else:
                st.write("Predicción del modelo:","Normal", prediction)

        

    
# Colocar el checkbox en la barra lateral
if st.sidebar.checkbox("Mostrar hiperparámetros del modelo"):
    st.write("#### Hiperparámetros del modelo")

    # Mostrar los hiperparámetros del modelo 1 (modelo de sklearn)
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
        st.dataframe(model1_params_df)

    # Mostrar los hiperparámetros del modelo 2 (modelo de red neuronal)
    if hasattr(model2, 'get_config'):
        st.write("##### Hiperparámetros del modelo de red neuronal (TensorFlow/Keras)")

        # Obtener los hiperparámetros de la red neuronal
        model2_config = model2.get_config()

        # Mostrar la configuración de la red neuronal en un formato legible
        model2_params = []
        for layer in model2.layers:
            layer_info = {
                "Capa": layer.__class__.__name__,  # Nombre de la capa (ej. Dense, Conv2D)
                "Hiperparámetros": layer.get_config()  # Obtiene la configuración de la capa
            }
            model2_params.append(layer_info)

        # Mostrar los parámetros del modelo en un DataFrame
        model2_params_df = pd.DataFrame(model2_params)
        st.dataframe(model2_params_df)

        # Obtener el learning rate
        if hasattr(model2, 'optimizer'):
            optimizer = model2.optimizer
            if hasattr(optimizer, 'lr'):  # Para versiones más antiguas de Keras
                learning_rate = optimizer.lr.numpy()
            elif hasattr(optimizer, 'learning_rate'):  # Para versiones más recientes de TensorFlow
                learning_rate = optimizer.learning_rate.numpy()

            st.write(f"##### Learning Rate: {learning_rate}")

        # Información adicional sobre las épocas, batch_size y otros parámetros de entrenamiento
        st.write("##### Información adicional:")
        st.write("Las épocas, el tamaño del batch y otros parámetros de entrenamiento no se almacenan directamente en el modelo. Es necesario que estos valores sean proporcionados de manera explícita.")
    else:
        st.write("El modelo no tiene el método get_config() disponible.")



# Continuación del flujo para predicción manual o por defecto
