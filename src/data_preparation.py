
import pandas as pd
import numpy as np
import os

INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-processed.csv'


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):

    # Carga del dataset
    df = pd.read_csv(input_path, sep=";")

    # Adaptar nombres de columnas
    df.columns = df.columns.str.replace(".", "_", regex=False)

    # Transformar los valores unknown en NaN
    df.replace('unknown', np.nan, inplace=True)

    # Se elimina la columna 'default' ya que tiene muchos valores unknown
    df.drop(columns=["default"], inplace=True, errors="ignore")

    # Se eliminan las filas que contienen valores nulos
    df.dropna(inplace=True)

    # Se eliminan las filas duplicadas
    df.drop_duplicates(inplace=True)

    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar el dataset procesado
    df.to_csv(output_path, index=False)

    return df.shape


if __name__ == "__main__":

    dimensiones = preprocess_data()

    # Crear directorio docs si no existe
    os.makedirs("docs", exist_ok=True)

    # Generar documentación de las transformaciones
    with open('docs/transformations.txt', "w", encoding="utf-8") as f:

        f.write("Transformaciones realizadas:\n")
        f.write("- Se reemplazaron los valores 'unknown' por NaN.\n")
        f.write("- Se eliminó la columna 'default' debido a la alta cantidad de valores unknown.\n")
        f.write("- Se eliminaron las filas con valores nulos.\n")
        f.write("- Se eliminaron las filas duplicadas.\n")
        f.write(f"- Cantidad de filas finales: {dimensiones[0]}\n")
        f.write(f"- Cantidad de columnas finales: {dimensiones[1]}\n")
