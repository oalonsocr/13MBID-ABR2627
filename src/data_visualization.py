import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def visualizar_datos(fuente: str = "data/raw/bank-additional-full.csv", 
                     salida: str = "docs/figure/"):
    """
    Genera una serie de graficos sobre los datos y los exporta
    """
    # Crear directorio  
    Path(salida).mkdir(parents=True, exist_ok=True)

    # leer los datos

    df = pd.read_csv(fuente, sep=";")

    # distribucion de las variables objetivo
    plt.figure(figsize=(6,4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo (suscripción al depósito)")
    plt.xlabel("¿Suscribió un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.savefig(f'{salida}/distribucion_target.png')
    plt.close()


# Grafico 2:  Distribucion del nivel educativo
    plt.figure(figsize=(8, 4))
    col = "education"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f'{salida}/distribucion_education.png')
    plt.close()


# Grafico 3:  Distribucion marital
    plt.figure(figsize=(8, 4))
    col = "marital"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f'{salida}/distribucion_marital.png')
    plt.close()


# Grafico 4:  Distribucion housing
    plt.figure(figsize=(8, 4))
    col = "housing"
    order = df[col].value_counts().index
    sns.countplot(y=col, data=df, order=order)
    plt.title(f"Distribución de {col}")
    plt.xlabel("Cantidad")
    plt.ylabel(col)
    plt.savefig(f'{salida}/distribucion_housing.png')
    plt.close()

if __name__ == "__main__":
    visualizar_datos()