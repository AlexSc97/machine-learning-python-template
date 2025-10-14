
# Importo las librerías que voy a necesitar.
# Pandas para manejar los datos del CSV.
import pandas as pd
# Sklearn para seleccionar las mejores características.
from sklearn.feature_selection import SelectKBest, f_regression


def cargar_datos(ruta_datos_crudos):
    """
    Esta función carga los datos crudos desde la ruta que le pasemos.
    Así no repito código si necesito cargar más archivos después.
    """
    print("1. Cargando los datos crudos...")
    return pd.read_csv(ruta_datos_crudos)


def procesar_y_limpiar_datos(df):
    """
    Aquí pasa toda la magia. Esta función toma el DataFrame crudo y lo deja
    listo para el modelo, siguiendo los pasos que descubrí en mi exploración.
    """
    print("2. Empezando la limpieza y transformación de los datos...")

    # Primero, me deshago de las columnas que no me sirven para predecir.
    # 'id', 'name', 'host_id', 'host_name' son identificadores.
    # 'last_review' y 'neighbourhood' las descarto porque ya tengo mejores datos.
    df.drop(['id', 'name', 'host_id', 'host_name', 'last_review', 'neighbourhood'], axis=1, inplace=True)

    # Relleno los valores que faltan en 'reviews_per_month' con un 0.
    # Esto es porque si no hay reseñas, las reseñas por mes son cero, tiene lógica.
    df['reviews_per_month'].fillna(0, inplace=True)

    # Ahora convierto las columnas de texto a números que el modelo entienda.
    # Uso get_dummies para crear columnas nuevas (0s y 1s) para los distritos y tipos de cuarto.
    df_procesado = pd.get_dummies(df, columns=['neighbourhood_group', 'room_type'], prefix=['dist', 'tipo'])

    print("--> Limpieza y transformación terminadas.")
    return df_procesado


def seleccionar_mejores_caracteristicas(df):
    """
    De todos los datos que tengo, esta función elige los mejores 8 para predecir el precio.
    Uso SelectKBest, que hace un test estadístico para calificarlas.
    """
    print("3. Seleccionando las características más importantes...")

    # Separo mi data en 'X' (las que predicen) y 'y' (la que quiero predecir, o sea, el precio).
    X = df.drop('price', axis=1)
    y = df['price']

    # Configuro el selector para que se quede con las 8 mejores.
    selector = SelectKBest(score_func=f_regression, k=8)
    selector.fit(X, y)

    # Guardo los nombres de las columnas que eligió.
    columnas_seleccionadas_mask = selector.get_support()
    nombres_columnas = X.columns[columnas_seleccionadas_mask]

    # Creo un nuevo DataFrame solo con el precio y las 8 mejores columnas.
    df_final = df[['price'] + list(nombres_columnas)]

    print(f"--> Características seleccionadas: {list(nombres_columnas)}")
    return df_final


def guardar_datos(df, ruta_guardado):
    """
    Esta función simplemente guarda el DataFrame final en la carpeta de procesados.
    Le pongo index=False para que no me guarde los números de fila en el CSV.
    """
    print(f"4. Guardando los datos finales en: {ruta_guardado}")
    df.to_csv(ruta_guardado, index=False)
    print("¡Listo! Proceso finalizado con éxito. ")


def main():
    """
    funcion principal que llamara a las demas funciones para realizar el proceso.
    """
    ruta_datos_crudos = "../data/raw/internal-link.csv"
    ruta_datos_procesados = "../data/processed/final_model_data.csv"

    # Paso 1: Cargar
    datos_crudos = cargar_datos(ruta_datos_crudos)

    # Paso 2: Procesar
    datos_procesados = procesar_y_limpiar_datos(datos_crudos.copy())

    # Paso 3: Seleccionar
    datos_finales_para_modelo = seleccionar_mejores_caracteristicas(datos_procesados)

    # Paso 4: Guardar
    guardar_datos(datos_finales_para_modelo, ruta_datos_procesados)


# Esto es para que el script se ejecute solo cuando lo llamo desde la terminal.
if __name__ == "__main__":
    main()
