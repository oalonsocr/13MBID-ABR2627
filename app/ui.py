
import streamlit as st
import requests
import pandas as pd
import json

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Bank Marketing - Machine Learning",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Bank Marketing Prediction")
st.markdown(
    """
    ### Sistema de predicción de aceptación de depósitos a plazo

    Aplicación de Machine Learning que permite evaluar la probabilidad
    de aceptación de una oferta bancaria a partir de las características
    del cliente y de las campañas de marketing.
    """
)

# ==========================================================
# 2. CONFIGURACIÓN DE LA API
# ==========================================================

st.sidebar.title("⚙️ Configuración")

st.sidebar.markdown("### Conexión con FastAPI")

API_BASE_URL = st.sidebar.text_input(
    "URL de la API",
    value="https://api-1mbid.onrender.com",
    placeholder="https://tu-api.onrender.com",
    help="Ingresa la URL base de tu API FastAPI."
).strip().rstrip("/")

# Permitir que el usuario ingrese la URL con o sin /predict
if API_BASE_URL.endswith("/predict"):
    API_BASE_URL = API_BASE_URL[:-8].rstrip("/")

API_URL = f"{API_BASE_URL}/predict"

st.sidebar.caption(f"Endpoint: {API_URL}")

# ----------------------------------------------------------
# PRUEBA DE CONEXIÓN
# ----------------------------------------------------------

if st.sidebar.button(
    "🔌 Probar conexión",
    use_container_width=True
):

    if not API_BASE_URL:
        st.sidebar.error("Ingresa una URL válida.")

    else:
        try:

            with st.spinner("Verificando conexión..."):

                response = requests.get(
                    f"{API_BASE_URL}/openapi.json",
                    timeout=30
                )

            response.raise_for_status()

            st.sidebar.success("✅ API conectada correctamente")

        except requests.exceptions.RequestException as error:

            st.sidebar.error(
                f"No se pudo conectar con la API: {error}"
            )

st.sidebar.divider()

st.sidebar.markdown(
    """
    ### Información del proyecto

    - **Modelo:** Gradient Boosting
    - **API:** FastAPI
    - **Interfaz:** Streamlit
    - **Dataset:** Bank Marketing
    - **Proyecto:** 13MBID-ABR2627
    """
)

# ==========================================================
# 3. FORMULARIO DE ENTRADA
# ==========================================================

st.subheader("📝 Información del cliente")

st.info(
    "Completa los siguientes campos para ejecutar una predicción "
    "utilizando el modelo entrenado."
)

with st.form("prediction_form"):

    # ======================================================
    # INFORMACIÓN PERSONAL
    # ======================================================

    st.markdown("### 👤 Información personal")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Edad",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    with col2:

        job = st.selectbox(
            "Ocupación",
            [
                "admin.",
                "blue-collar",
                "entrepreneur",
                "housemaid",
                "management",
                "retired",
                "self-employed",
                "services",
                "student",
                "technician",
                "unemployed",
                "unknown"
            ]
        )

    with col3:

        marital = st.selectbox(
            "Estado civil",
            [
                "married",
                "single",
                "divorced",
                "unknown"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        education = st.selectbox(
            "Nivel educativo",
            [
                "university.degree",
                "high.school",
                "basic.9y",
                "basic.6y",
                "basic.4y",
                "professional.course",
                "illiterate",
                "unknown"
            ]
        )

    with col2:

        housing = st.selectbox(
            "¿Tiene préstamo hipotecario?",
            ["yes", "no", "unknown"]
        )

    col1, col2 = st.columns(2)

    with col1:

        loan = st.selectbox(
            "¿Tiene préstamo personal?",
            ["no", "yes", "unknown"]
        )

    with col2:

        contact = st.selectbox(
            "Tipo de contacto",
            ["cellular", "telephone"]
        )

    # ======================================================
    # INFORMACIÓN DE LA CAMPAÑA
    # ======================================================

    st.divider()

    st.markdown("### 📞 Información de la campaña")

    col1, col2, col3 = st.columns(3)

    with col1:

        month = st.selectbox(
            "Mes del último contacto",
            [
                "jan", "feb", "mar", "apr",
                "may", "jun", "jul", "aug",
                "sep", "oct", "nov", "dec"
            ],
            index=4
        )

    with col2:

        day_of_week = st.selectbox(
            "Día de la semana",
            ["mon", "tue", "wed", "thu", "fri"]
        )

    with col3:

        duration = st.number_input(
            "Duración de la llamada (segundos)",
            min_value=0,
            value=200,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        campaign = st.number_input(
            "Contactos en la campaña actual",
            min_value=1,
            value=2,
            step=1
        )

    with col2:

        pdays = st.number_input(
            "Días desde el contacto anterior",
            min_value=0,
            value=999,
            step=1,
            help="999 indica que el cliente no fue contactado anteriormente."
        )

    with col3:

        previous = st.number_input(
            "Contactos en campañas anteriores",
            min_value=0,
            value=0,
            step=1
        )

    poutcome = st.selectbox(
        "Resultado de la campaña anterior",
        [
            "nonexistent",
            "failure",
            "success"
        ]
    )

    # ======================================================
    # INDICADORES ECONÓMICOS
    # ======================================================

    st.divider()

    st.markdown("### 📈 Indicadores económicos")

    col1, col2, col3 = st.columns(3)

    with col1:

        emp_var_rate = st.number_input(
            "Tasa de variación del empleo",
            value=1.1,
            format="%.3f"
        )

    with col2:

        cons_price_idx = st.number_input(
            "Índice de precios al consumidor",
            value=93.994,
            format="%.3f"
        )

    with col3:

        cons_conf_idx = st.number_input(
            "Índice de confianza del consumidor",
            value=-36.4,
            format="%.3f"
        )

    col1, col2 = st.columns(2)

    with col1:

        euribor3m = st.number_input(
            "Tasa Euribor 3 meses",
            value=4.857,
            format="%.3f"
        )

    with col2:

        nr_employed = st.number_input(
            "Número de empleados",
            value=5191.0,
            format="%.3f"
        )

    # ======================================================
    # BOTÓN DE PREDICCIÓN
    # ======================================================

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Ejecutar predicción",
        use_container_width=True
    )


# ==========================================================
# 4. EJECUCIÓN DE LA PREDICCIÓN
# ==========================================================

if submitted:

    if not API_BASE_URL:

        st.error("Debes ingresar la URL de la API.")

        st.stop()

    # ------------------------------------------------------
    # CONSTRUCCIÓN DEL PAYLOAD
    # ------------------------------------------------------

    payload = {

        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed

    }

    # ------------------------------------------------------
    # SOLICITUD HTTP
    # ------------------------------------------------------

    try:

        with st.spinner(
            "Procesando información con el modelo de Machine Learning..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            resultado = response.json()

        
        # ==========================================================
        # 5. RESULTADOS DE LA PREDICCIÓN
        # ==========================================================

        st.divider()
        st.subheader("🎯 Resultado de la predicción")

        prediction = resultado.get("prediction")
        probabilities = resultado.get("probability", {})
        model_info = resultado.get("model_info", {})

        # ----------------------------------------------------------
        # INTERPRETACIÓN DE LA PREDICCIÓN
        # ----------------------------------------------------------

        prediction_text = str(prediction).lower()

        if prediction_text == "1":

            st.success(
                "El modelo predice que el cliente SÍ aceptaría "
                "la oferta de depósito a plazo."
            )

        elif prediction_text == "0":

            st.warning(
                "El modelo predice que el cliente NO aceptaría "
                "la oferta de depósito a plazo."
            )

        else:

            st.info(f"Predicción obtenida: {prediction}")

        # ----------------------------------------------------------
        # PROBABILIDADES
        # ----------------------------------------------------------

        st.markdown("### 📊 Probabilidades por clase")

        if probabilities:

            for clase, valor in probabilities.items():

                porcentaje = float(valor) * 100

                if str(clase) == "1":
                    nombre_clase = "Aceptaría la oferta"
                else:
                    nombre_clase = "No aceptaría la oferta"

                st.markdown(
                    f"**Clase {clase} - {nombre_clase}: {porcentaje:.2f}%**"
                )

                st.progress(
                    max(0.0, min(1.0, float(valor)))
                )

        else:

            st.warning("No se recibieron probabilidades.")

        # ----------------------------------------------------------
        # INFORMACIÓN DEL MODELO
        # ----------------------------------------------------------

        st.divider()

        st.markdown("### ⚙️ Información del modelo")

        if isinstance(model_info, dict):

            for clave, valor in model_info.items():

                nombre = clave.replace("_", " ").title()

                st.markdown(f"- **{nombre}:** {valor}")

        else:

            st.write(str(model_info))

        # ----------------------------------------------------------
        # RESPUESTA COMPLETA DE FASTAPI
        # ----------------------------------------------------------

        st.divider()

        st.markdown("### 🔍 Respuesta completa de la API")

        st.code(
            json.dumps(
                resultado,
                indent=4,
                ensure_ascii=False
            ),
            language="json"
        )


    # ======================================================
    # 6. MANEJO DE ERRORES
    # ======================================================

    except requests.exceptions.Timeout:

        st.error(
            "La API tardó demasiado en responder. "
            "Intenta nuevamente en unos segundos."
        )

    except requests.exceptions.HTTPError as error:

        st.error(
            f"Error HTTP: {error}"
        )

        if response is not None:
            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "No se pudo establecer conexión con FastAPI. "
            "Verifica la URL y el estado del servicio."
        )

    except requests.exceptions.RequestException as error:

        st.error(
            f"Error en la solicitud: {error}"
        )

    except (ValueError, TypeError) as error:

        st.error(
            f"Error al interpretar la respuesta de la API: {error}"
        )

    except Exception as error:

        st.error(
            f"Se produjo un error inesperado: {error}"
        )


# ==========================================================
# 7. PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    "Proyecto 13MBID-ABR2627 | Machine Learning | "
    "Bank Marketing Prediction"
)
