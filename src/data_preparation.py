
import pandas as pd
import numpy as np
import os

INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-processed.csv'


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):

    # Cargar dataset original
    df = pd.read_csv(input_path, sep=";")

    filas_iniciales = len(df)
    transformaciones = []

    # 1. Adaptar nombres de columnas
    df.columns = df.columns.str.replace(".", "_", regex=False)
    transformaciones.append("Se reemplazaron los puntos por guiones bajos en los nombres de columnas.")

    # 2. Normalizar valores de texto
    columnas_texto = df.select_dtypes(include=["object"]).columns

    for columna in columnas_texto:
        df[columna] = df[columna].str.strip().str.lower()

    transformaciones.append("Se eliminaron espacios innecesarios y se unificaron los textos en minúsculas.")

    # 3. Eliminar columna default
    df.drop(columns=["default"], inplace=True, errors="ignore")
    transformaciones.append("Se eliminó la columna default debido a su alta proporción de valores unknown.")

    # 4. Conservar unknown como categoría
    # No se eliminan automáticamente los registros que contienen unknown.
    transformaciones.append("Se conservaron los valores unknown en las variables categóricas para evitar pérdida de información.")

    # 5. Eliminar registros duplicados
    duplicados = df.duplicated().sum()
    df.drop_duplicates(inplace=True)


    transformaciones.append(f"Se eliminaron {duplicados} registros duplicados.")

    map = {'yes':1, 'no':0}
    df['y']= df['y'].map(map)

    # 6. Validar rangos de variables numéricas
    # Se identifican valores inválidos sin eliminarlos automáticamente.
    validaciones = {
        "age": df["age"].between(17, 100),
        "campaign": df["campaign"] >= 1,
        "duration": df["duration"] >= 0,
        "previous": df["previous"] >= 0,
        "pdays": df["pdays"].between(0, 999)
    }

    for columna, condicion in validaciones.items():
        if not condicion.all():
            raise ValueError(
                f"Se encontraron {int((~condicion).sum())} valores fuera del rango esperado en {columna}."
            )

    transformaciones.append("Se validaron los rangos de edad, duración, campaña, contactos anteriores y pdays.")

    # 7. Crear variable de grupo etario
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"]
    )

    transformaciones.append("Se creó la variable derivada age_group para agrupar las edades.")

    # 8. Crear indicador de contacto previo
    df["previously_contacted"] = np.where(
        df["pdays"] == 999, 0, 1
    )

    transformaciones.append("Se creó previously_contacted para identificar clientes contactados anteriormente.")

    # 9. Agrupar cantidad de contactos de campaña
    df["campaign_group"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, float("inf")],
        labels=["1", "2-3", "4-5", "6+"]
    )

    transformaciones.append("Se creó campaign_group para categorizar la cantidad de contactos de campaña.")

    # 10. Convertir variables categóricas
    columnas_categoricas = [
        "job", "marital", "education", "housing",
        "loan", "contact", "month", "day_of_week",
        "poutcome", "y", "age_group", "campaign_group"
    ]

    for columna in columnas_categoricas:
        if columna in df.columns:
            df[columna] = df[columna].astype("category")

    transformaciones.append("Se convirtieron las variables categóricas al tipo category.")

    # 11. Crear directorio de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 12. Guardar dataset procesado
    df.to_csv(output_path, index=False)

    filas_finales = len(df)
    columnas_finales = len(df.columns)

    return {
        "filas_iniciales": filas_iniciales,
        "filas_finales": filas_finales,
        "filas_eliminadas": filas_iniciales - filas_finales,
        "columnas_finales": columnas_finales,
        "transformaciones": transformaciones
    }


if __name__ == "__main__":

    resultado = preprocess_data()

    # Crear directorio docs
    os.makedirs("docs", exist_ok=True)

    # Generar documentación de las transformaciones
    with open("docs/transformations.txt", "w", encoding="utf-8") as f:

        f.write("TRANSFORMACIONES REALIZADAS\n")
        f.write("===========================\n\n")

        for i, transformacion in enumerate(resultado["transformaciones"], 1):
            f.write(f"{i}. {transformacion}\n")

        f.write("\nRESUMEN DEL PROCESAMIENTO\n")
        f.write("========================\n")

        f.write(f"Filas iniciales: {resultado['filas_iniciales']}\n")
        f.write(f"Filas finales: {resultado['filas_finales']}\n")
        f.write(f"Filas eliminadas: {resultado['filas_eliminadas']}\n")
        f.write(f"Columnas finales: {resultado['columnas_finales']}\n")

