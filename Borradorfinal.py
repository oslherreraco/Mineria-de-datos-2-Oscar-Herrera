# Mostrar los hiperparámetros de los modelos
if st.checkbox("Mostrar hiperparámetros del modelo"):
    st.write("#### Hiperparámetros del modelo")

    # Mostrar hiperparámetros del modelo 1 (sklearn) si está cargado
    if hasattr(model1, 'get_params'):
        model1_params = model1.get_params()

        # Convertir los hiperparámetros a un formato adecuado para una tabla
        model1_params_table = [(key, value) for key, value in model1_params.items()]

        # Reemplazar <NA> o None por un guion o valor vacío
        cleaned_model1_params = [
            (key, value if value is not None and value != "<NA>" else "-") 
            for key, value in model1_params_table
        ]

        # Mostrar los hiperparámetros del modelo 1 (sklearn) en una tabla
        st.write("#### Hiperparámetros del modelo 1 (sklearn):")
        model1_params_df = pd.DataFrame(cleaned_model1_params, columns=["Hiperparámetro", "Valor"])
        st.dataframe(model1_params_df)

    # Mostrar hiperparámetros del modelo 2 (red neuronal) si está cargado
    if hasattr(model2, 'get_config'):
        model2_config = model2.get_config()
        model2_params = model2_config[0]['config']

        # Extraer y mostrar los parámetros de la red neuronal
        model2_params_table = [(key, value) for key, value in model2_params.items()]

        # Mostrar los hiperparámetros del modelo 2 (red neuronal) en una tabla
        st.write("#### Hiperparámetros del modelo 2 (red neuronal):")
        model2_params_df = pd.DataFrame(model2_params_table, columns=["Hiperparámetro", "Valor"])
        st.dataframe(model2_params_df)

    # Si no hay modelos cargados, mostrar un mensaje de advertencia
    if not hasattr(model1, 'get_params') and not hasattr(model2, 'get_config'):
        st.warning("No se encontraron modelos cargados para mostrar los hiperparámetros.")
