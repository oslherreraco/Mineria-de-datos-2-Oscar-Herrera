# Cargar y mostrar los hiperparámetros de los modelos
if st.checkbox("Mostrar hiperparámetros del modelo"):
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
        for layer in model2_config:
            layer_info = {
                "Capa": layer['class_name'],
                "Hiperparámetros": layer['config']
            }
            model2_params.append(layer_info)

        model2_params_df = pd.DataFrame(model2_params)

        # Mostrar la tabla con los parámetros de la red neuronal
        st.dataframe(model2_params_df)

# Continuación del flujo para predicción manual o por defecto
if selected_column == 'Manual':
    st.write("### Ingresar datos manualmente para predicción")
    # Aquí debes agregar los widgets para que el usuario ingrese los datos manualmente.
    # Dependiendo de los valores que necesitas para la red neuronal, deberías crear inputs para ellos.
    # Ejemplo:
    input_data = []
    input_data.append(st.number_input("Edad", min_value=0))
    input_data.append(st.number_input("Peso", min_value=0))
    # Aquí añadir los campos adicionales necesarios para completar las características de entrada de la red neuronal.

    # Cuando se ingresen los datos, puedes hacer la predicción de la siguiente manera:
    if st.button("Realizar Predicción"):
        input_data = np.array(input_data).reshape(1, -1)
        prediction = model2.predict(input_data)
        st.write("Predicción de la red neuronal:", prediction)


    
