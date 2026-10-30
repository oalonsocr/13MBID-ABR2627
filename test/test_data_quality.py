import pandas as pd
import great_expectations
from pandera.pandas import DataFrameSchema, Column
import pytest
from pathlib import Path

@pytest.fixture
def datos_banco():
    """
    Fixture para la carga de datos del banco desde un archivo csv
    """

    ruta_proyecto = Path(__file__).resolve().parents[1]
    ruta_archivo = ruta_proyecto / "data" / "raw" / "bank-additional-full.csv"
    df = pd.read_csv(ruta_archivo, sep=";")

    return df

def test_esquema(datos_banco):
    """
    Test de esquema para el Dataframe de datos del banco
    """

    df = datos_banco

    esquema = DataFrameSchema({
        "age": Column(int, nullable=False),
        "job": Column(str, nullable=False),
        "marital": Column(str, nullable=False),
        "education": Column(str, nullable=False),
        "default": Column(str, nullable=False),
        "housing": Column(str, nullable=False),
        "loan": Column(str, nullable=False),
        "contact": Column(str, nullable=False),
        "month": Column(str, nullable=False),
        "day_of_week": Column(str, nullable=False),
        "duration": Column(int, nullable=False),
        "campaign": Column(int, nullable=False),
        "pdays": Column(int, nullable=False),
        "previous": Column(int, nullable=False),
        "poutcome": Column(str, nullable=False),
        "emp.var.rate": Column(float, nullable=False),
        "cons.price.idx": Column(float, nullable=False),
        "cons.conf.idx": Column(float, nullable=False),
        "euribor3m": Column(float, nullable=False),
        "nr.employed": Column(float, nullable=False),
        "y": Column(str, nullable=False)
    }, strict=True)

    esquema.validate(df)



def test_basico(datos_banco):
    """
    Test basico para verificar que el dataframe  de datos_banco
    no esta vacio y contiene columnas esperadas.
    """


    df = datos_banco

    # verificar que el df no este vacio
    assert not df.empty, "El dataFrame esta vacio"

    # verificar nulos
    assert df.isnull().sum().sum() == 0, "El dataFrame contiene valores nulos"

    # verificar filas duplicadas
    #assert df.duplicated().sum() == 0 , "El dataFrame contiene filas duplicadas"

    # verificar cantidad de columnas
    assert df.shape[1] == 21, f'El dataFrame deberia tener 21 columnas , pero tiene {df.shape[1]}'



if __name__ == "__main__":
    try:
        test_esquema(datos_banco())
        test_basico(datos_banco())
        print("Todos los test pasaron exitosamente")
        with open("docs/test_results/test_results.txt", "w") as f:
            f.write("Todos los tests pasaron exitosamente.\n")
    
    except AssertionError as e:
        print(f'Test fallido: {e}')
        with open("docs/test_results/test_results.txt", "w") as f:
            f.write(f'Test fallido: {e}\n')