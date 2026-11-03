
"""
Script para entrenar y evaluar modelos de clasificación
utilizando Random Undersampling y registrando experimentos en MLflow.
"""

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score
)
from mlflow.models import infer_signature

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.naive_bayes import GaussianNB


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "processed" / "bank-processed.csv"

MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_PATH = MODEL_DIR / "best_model.joblib"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

EXPERIMENT_NAME = "Proyecto 13MBID-ABR2627 - Experimentacion original"


# =========================================================
# 1. CONFIGURACIÓN DE MLFLOW
# =========================================================

def configure_mlflow():
    """
    Configura el almacenamiento de experimentos en MLflow.
    """

    mlflow_db = PROJECT_DIR / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{mlflow_db.as_posix()}"
    )

    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Tracking URI:", mlflow.get_tracking_uri())
    print("Experimento:", EXPERIMENT_NAME)


# =========================================================
# 2. CARGA DE DATOS
# =========================================================

def load_data(path=DATA_PATH):
    """
    Carga el dataset procesado y separa las variables
    predictoras de la variable objetivo.
    """

    df = pd.read_csv(path)

    if "y" not in df.columns:
        raise ValueError("No se encontró la columna objetivo 'y'.")

    X = df.drop(columns=["y"])
    y = df["y"]

    return X, y


# =========================================================
# 3. DIVISIÓN DE DATOS
# =========================================================

def split_data(X, y):
    """
    Divide los datos en entrenamiento y prueba,
    manteniendo la distribución de clases.
    """

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# =========================================================
# 4. PREPROCESAMIENTO
# =========================================================

def create_preprocessor(X):
    """
    Configura el preprocesamiento de variables numéricas
    y categóricas.
    """

    numerical_columns = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ("RobustScaler", RobustScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("OneHotEncoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, numerical_columns),
            ("cat_pipeline", cat_pipeline, categorical_columns)
        ]
    ).set_output(transform="pandas")

    return preprocessor


# =========================================================
# 5. BALANCEO DE DATOS
# =========================================================

def balance_data(X, y):
    """
    Aplica Random Undersampling al conjunto de entrenamiento.
    Reduce cada clase hasta el tamaño de la clase minoritaria.
    """

    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    train_data = X.copy()
    train_data["target"] = y.to_numpy()

    class_counts = train_data["target"].value_counts()
    min_count = class_counts.min()

    balanced_classes = []

    for class_label in class_counts.index:

        class_data = train_data[
            train_data["target"] == class_label
        ]

        class_balanced = resample(
            class_data,
            replace=False,
            n_samples=min_count,
            random_state=RANDOM_STATE
        )

        balanced_classes.append(class_balanced)

    balanced_data = pd.concat(
        balanced_classes
    ).sample(
        frac=1,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    X_balanced = balanced_data.drop(columns=["target"])
    y_balanced = balanced_data["target"]

    return X_balanced, y_balanced


# =========================================================
# 6. VALIDACIÓN CRUZADA
# =========================================================

def evaluate_cross_validation(model, X_train, y_train):
    """
    Realiza validación cruzada estratificada.

    El preprocesamiento y el balanceo se ajustan dentro
    de cada fold para evitar filtración de información.
    """

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    metrics = {
        "f1": [],
        "recall": [],
        "precision": [],
        "accuracy": []
    }

    for train_idx, valid_idx in cv.split(X_train, y_train):

        X_fold_train = X_train.iloc[train_idx]
        X_fold_valid = X_train.iloc[valid_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_valid = y_train.iloc[valid_idx]

        # Preprocesamiento dentro del fold
        preprocessor = create_preprocessor(X_fold_train)

        X_fold_train_prep = preprocessor.fit_transform(
            X_fold_train
        )

        X_fold_valid_prep = preprocessor.transform(
            X_fold_valid
        )

        # Balanceo únicamente del fold de entrenamiento
        X_fold_balanced, y_fold_balanced = balance_data(
            X_fold_train_prep,
            y_fold_train
        )

        # Entrenamiento
        fold_model = clone(model)

        fold_model.fit(
            X_fold_balanced,
            y_fold_balanced
        )

        # Predicción sobre el fold de validación original
        y_pred = fold_model.predict(X_fold_valid_prep)

        metrics["f1"].append(
            f1_score(y_fold_valid, y_pred, zero_division=0)
        )

        metrics["recall"].append(
            recall_score(y_fold_valid, y_pred, zero_division=0)
        )

        metrics["precision"].append(
            precision_score(y_fold_valid, y_pred, zero_division=0)
        )

        metrics["accuracy"].append(
            accuracy_score(y_fold_valid, y_pred)
        )

    results = {}

    for metric_name, scores in metrics.items():

        results[f"cv_{metric_name}_mean"] = float(
            pd.Series(scores).mean()
        )

        results[f"cv_{metric_name}_std"] = float(
            pd.Series(scores).std(ddof=0)
        )

    return results


# =========================================================
# 7. REGISTRO EN MLFLOW
# =========================================================

def log_experiment(model, model_name, pipeline, X_test, y_pred,
                   cv_metrics, test_metrics):
    """
    Registra parámetros, métricas y modelo en MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("balancing_method", "Random Undersampling")
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Registrar parámetros del modelo con un prefijo
        model_params = {
            f"model_{key}": value
            for key, value in model.get_params().items()
        }

        mlflow.log_params(model_params)

        mlflow.log_metrics(cv_metrics)
        mlflow.log_metrics(test_metrics)

        signature = infer_signature(X_test, y_pred)

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        run_id = mlflow.active_run().info.run_id

        print(
            f"Modelo {model_name} registrado. "
            f"Run ID: {run_id}"
        )


# =========================================================
# 8. ENTRENAMIENTO Y EVALUACIÓN
# =========================================================

def train_and_evaluate_model(
    model,
    model_name,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Ejecuta validación cruzada, entrenamiento final,
    evaluación sobre test y registro en MLflow.
    """

    print(f"\nEntrenando: {model_name}")

    # Validación cruzada
    cv_metrics = evaluate_cross_validation(
        model,
        X_train,
        y_train
    )

    # Preprocesamiento final sobre todo el entrenamiento
    preprocessor = create_preprocessor(X_train)

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Balanceo del entrenamiento
    X_train_balanced, y_train_balanced = balance_data(
        X_train_prep,
        y_train
    )

    # Entrenamiento final
    final_model = clone(model)

    final_model.fit(
        X_train_balanced,
        y_train_balanced
    )

    # Pipeline para conservar preprocesamiento y modelo
    final_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", final_model)
    ])

    # Evaluación sobre test sin balancear
    y_pred = final_pipeline.predict(X_test)

    test_metrics = {
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_pred)
    }

    # Registro en MLflow
    log_experiment(
        final_model,
        model_name,
        final_pipeline,
        X_test,
        y_pred,
        cv_metrics,
        test_metrics
    )

    results = {
        **cv_metrics,
        **test_metrics
    }

    return final_pipeline, results


# =========================================================
# 9. DEFINICIÓN DE MODELOS
# =========================================================

def get_models():
    """
    Define los siete algoritmos utilizados
    durante la experimentación.
    """

    return {
        "Logistic Regression": LogisticRegression(
            C=1,
            solver="liblinear",
            random_state=1,
            max_iter=100,
            tol=0.000000001
        ),

        "Linear SVC": LinearSVC(
            max_iter=10000,
            tol=0.001,
            random_state=42
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7
        ),

        "Decision Tree Classifier": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),

        "Gaussian Naive Bayes": GaussianNB()
    }


# =========================================================
# 10. GUARDADO DE RESULTADOS
# =========================================================

def save_results(results, trained_models):
    """
    Guarda la comparación de modelos y el modelo seleccionado.

    La selección se realiza mediante el F1 promedio
    de validación cruzada, no mediante el resultado de test.
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results).T

    df_results = df_results.sort_values(
        by="cv_f1_mean",
        ascending=False
    )

    df_results.to_csv(COMPARISON_PATH, index=True)

    best_model_name = df_results.index[0]
    best_pipeline = trained_models[best_model_name]

    joblib.dump(best_pipeline, MODEL_PATH)

    best_metrics = {
        "best_model": best_model_name,
        "selection_metric": "cv_f1_mean",
        "metrics": {
            key: float(value)
            for key, value in df_results.loc[best_model_name].items()
        }
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(best_metrics, file, indent=4, ensure_ascii=False)

    print("\nComparación de modelos:")
    print(df_results.round(4))

    print("\nModelo seleccionado:", best_model_name)
    print("Modelo guardado en:", MODEL_PATH)


# =========================================================
# 11. FUNCIÓN PRINCIPAL
# =========================================================

def main():

    # Configuración
    configure_mlflow()

    # Carga de datos
    X, y = load_data()

    # División de datos
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Datos de entrenamiento:", X_train.shape)
    print("Datos de prueba:", X_test.shape)

    # Modelos
    models = get_models()

    results = {}
    trained_models = {}

    # Experimentación
    for model_name, model in models.items():

        trained_pipeline, metrics = train_and_evaluate_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[model_name] = metrics
        trained_models[model_name] = trained_pipeline

    # Resultados finales
    save_results(results, trained_models)


if __name__ == "__main__":
    main()
