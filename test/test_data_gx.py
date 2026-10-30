
import pandas as pd


def test_great_expectations():
    """
    Test para verificar que los datos cumplen
    con las expectativas definidas.
    """

    df = pd.read_csv(
        "data/raw/bank-additional-full.csv",
        sep=";"
    )

    results = {
        "success": True,
        "expectations": [],
        "statistics": {
            "success_count": 0,
            "total_count": 0
        }
    }

    # Validaciones a verificar sobre los datos

    add_expectation(
    results,
    "age_range",
    df["age"].between(17, 100).all(),
    "La columna 'age' no está en el rango esperado (17 - 100)"
    )

    add_expectation(
        results,
        "target_values",
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores no permitidos"
    )

    # Expectation adicional 1
    add_expectation(
        results,
        "campaign_range",
        (df["campaign"] >= 1).all(),
        "La columna 'campaign' contiene valores menores a 1"
    )

    # Expectation adicional 2
    add_expectation(
        results,
        "pdays_range",
        df["pdays"].between(0, 999).all(),
        "La columna 'pdays' contiene valores fuera del rango esperado (0 - 999)"
    )

    print(results)

    assert results["success"], "Algunas expectativas fallaron."


def add_expectation(results, expectation_name, condition, message=""):

    results["statistics"]["total_count"] += 1

    if condition:
        results["statistics"]["success_count"] += 1

        results["expectations"].append({
            "expectation": expectation_name,
            "success": True
        })

    else:
        results["success"] = False

        results["expectations"].append({
            "expectation": expectation_name,
            "success": False,
            "message": message
        })
