
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================
# CONFIGURACION DE RUTAS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    PROJECT_ROOT / "config" / "model_metrics_baseline.json"
)

COMPARISON_PATH = (
    PROJECT_ROOT / "reports" / "model_comparison.csv"
)

REPORT_PATH = (
    PROJECT_ROOT / "reports" / "training_regression.json"
)


# ============================================================
# CARGA DE CONFIGURACION Y RESULTADOS
# ============================================================

def load_baseline():
    """Carga las métricas de referencia del modelo."""

    if not BASELINE_PATH.exists():
        pytest.fail(
            f"No se encontró la línea base: {BASELINE_PATH}"
        )

    with open(BASELINE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_model_comparison():
    """Carga los resultados de la experimentación."""

    if not COMPARISON_PATH.exists():
        pytest.fail(
            f"No se encontró el archivo de comparación: "
            f"{COMPARISON_PATH}. Ejecuta primero el entrenamiento."
        )

    df = pd.read_csv(COMPARISON_PATH)

    # Identificar la columna que contiene el nombre del modelo.
    if "model" in df.columns:
        pass

    elif "Modelo" in df.columns:
        df = df.rename(columns={"Modelo": "model"})

    else:
        # Compatibilidad con CSV exportados con índice de pandas.
        first_column = df.columns[0]
        df = df.rename(columns={first_column: "model"})

    return df


# ============================================================
# VALIDACION DE METRICAS
# ============================================================

def validate_training_metrics():
    """
    Compara las métricas del modelo actual contra la línea base.

    Retorna un reporte con los resultados de la validación.
    """

    baseline = load_baseline()
    comparison = load_model_comparison()

    selected_model = baseline["selected_model"]
    baseline_metrics = baseline["metrics"]
    tolerances = baseline["tolerances"]

    # Buscar el modelo seleccionado en la experimentación.
    model_result = comparison[
        comparison["model"].astype(str).str.strip()
        == selected_model
    ]

    if model_result.empty:
        pytest.fail(
            f"No se encontró el modelo '{selected_model}' "
            f"en {COMPARISON_PATH.name}."
        )

    model_result = model_result.iloc[0]

    metrics_to_check = [
        "test_f1",
        "test_recall",
        "test_precision",
        "test_accuracy"
    ]

    results = {}
    errors = []

    for metric in metrics_to_check:

        if metric not in comparison.columns:
            errors.append(
                f"No existe la métrica '{metric}' en el CSV."
            )
            continue

        if metric not in baseline_metrics:
            errors.append(
                f"No existe '{metric}' en la línea base."
            )
            continue

        if metric not in tolerances:
            errors.append(
                f"No existe tolerancia para '{metric}'."
            )
            continue

        current_value = float(model_result[metric])
        baseline_value = float(baseline_metrics[metric])
        tolerance = float(tolerances[metric])

        if not np.isfinite(current_value):
            errors.append(
                f"La métrica {metric} no tiene un valor válido."
            )
            continue

        if not 0 <= current_value <= 1:
            errors.append(
                f"La métrica {metric} está fuera del rango [0, 1]."
            )
            continue

        minimum_acceptable = baseline_value - tolerance
        difference = current_value - baseline_value

        passed = current_value >= minimum_acceptable

        results[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "difference": round(difference, 4),
            "minimum_acceptable": round(
                minimum_acceptable, 4
            ),
            "tolerance": tolerance,
            "status": "PASSED" if passed else "FAILED"
        }

        if not passed:
            errors.append(
                f"Regresión en {metric}: "
                f"actual={current_value:.4f}, "
                f"mínimo aceptable={minimum_acceptable:.4f}."
            )

    report = {
        "selected_model": selected_model,
        "baseline_date": baseline.get("baseline_date"),
        "status": "FAILED" if errors else "PASSED",
        "metrics": results,
        "errors": errors
    }

    # Guardar el resultado de la prueba.
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    return report


# ============================================================
# TEST DE REGRESION DEL ENTRENAMIENTO
# ============================================================

def test_training_metrics_regression():

    report = validate_training_metrics()

    print("\n====================================")
    print("VALIDACION DE REGRESION DEL MODELO")
    print("====================================")

    print(f"Modelo: {report['selected_model']}")
    print(f"Estado: {report['status']}")

    for metric, values in report["metrics"].items():

        print(
            f"{metric}: "
            f"baseline={values['baseline']:.4f} | "
            f"actual={values['current']:.4f} | "
            f"estado={values['status']}"
        )

    assert report["status"] == "PASSED", (
        "Se detectó una regresión en el entrenamiento: "
        + " | ".join(report["errors"])
    )
