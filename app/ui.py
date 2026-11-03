
import streamlit as st
import requests
import pandas as pd

# ==========================================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================================

st.set_page_config(
    page_title="Bank Marketing - Predicción ML",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sistema de Predicción - Bank Marketing")
st.markdown(
    """
    Aplicación de Machine Learning para predecir si un cliente
    aceptará una oferta de depósito a plazo.
    """
)

# ==========================================================
# 2. CONFIGURACIÓN DE LA API
# ==========================================================

st.sidebar.header("⚙️ Configuración de conexión")

API_BASE_URL = st.sidebar.text_input(
    "URL de la API FastAPI",
    value="https://api-1mbid.onrender.com",
    placeholder="https://tu-api.onrender.com",
    help="Ingresa la URL pública o local de tu API."
).strip().rstrip("/")

# Evitar duplicar /predict si el usuario pega la URL completa
if API_BASE_URL.endswith("/predict"):
    API_BASE_URL = API_BASE_URL[:-8].rstrip("/")

API_URL = f"{API_BASE_URL}/predict"

st.sidebar.caption(f"Endpoint: {API_URL}")

if st.sidebar.button("🔌 Probar conexión", use_container_width=True):

    if not API_BASE_URL:
        st.sidebar.error("Ingresa una URL válida.")
    else:
        try:
            with st.spinner("Conectando con FastAPI..."):
                response = requests.get(
                    API_BASE_URL,
                    timeout=30
                )

            response.raise_for_status()

            st.sidebar.success("✅ API disponible")

        except requests.exceptions.RequestException as error:
            st.sidebar.error(
                f"❌ No se pudo conectar con la API: {error}"
            )

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Información del proyecto**

    - Modelo: Gradient Boosting
    - Framework API: FastAPI
    - Interfaz: Streamlit
    - Dataset: Bank Marketing
    """
)

# ==========================================================
# 3. FORMULARIO DE ENTRADA
# ==========================================================

st.subheader("📝 Datos del cliente")

st.info(
    "Completa los datos del cliente para obtener una predicción "
    "del modelo entrenado."
)

with st.form("prediction_form"):

    # ------------------------------------------------------
    # VARIABLES PERSONALES
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # VARIABLES DE CAMPAÑA
    # ------------------------------------------------------

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
            step=1,
            help="Duración de la llamada en segundos."
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        campaign = st.number_input(
            "Número de contactos en esta campaña",
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
            help="999 significa que no hubo contacto previo."
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
        ["nonexistent", "failure", "success"]
    )

    # ------------------------------------------------------
    # VARIABLES ECONÓMICAS
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # BOTÓN DE PREDICCIÓN
    # ------------------------------------------------------

    st.divider()

    submitted = st.form_submit_button(
        "🚀 Ejecutar predicción",
        use_container_width=True
    )


# ==========================================================
# 4. CONSTRUCCIÓN DEL PAYLOAD Y PREDICCIÓN
# ==========================================================

if submitted:

    if not API_BASE_URL:
        st.error("Debes configurar la URL de la API.")
        st.stop()

    # Los 19 campos originales que espera el modelo
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

    try:

        with st.spinner(
            "El modelo está procesando los datos..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            resultado = response.json()

        # --------------------------------------------------
        # 5. MOSTRAR RESULTADOS
        # --------------------------------------------------

        st.divider()
        st.subheader("🎯 Resultado de la predicción")

        prediction = resultado.get("prediction")
        probabilities = resultado.get("probability", {})

        # Interpretación del resultado
        prediction_text = str(prediction).lower()

        if prediction_text in ["1", "yes", "true"]:
            st.success(
                "El modelo predice que el cliente SÍ aceptaría "
                "la oferta de depósito a plazo."
            )

        elif prediction_text in ["0", "no", "false"]:
            st.warning(
                "El modelo predice que el cliente NO aceptaría "
                "la oferta de depósito a plazo."
            )

        else:
            st.info(f"Predicción del modelo: {prediction}")

        # Mostrar probabilidades
        if probabilities:

            st.markdown("### 📊 Probabilidades por clase")

            probability_data = pd.DataFrame(
                {
                    "Clase": list(probabilities.keys()),
                    "Probabilidad": [
                        float(value) * 100
                        for value in probabilities.values()
                    ]
                }
            )

            col1, col2 = st.columns(2)

            for index, (_, row) in enumerate(
                probability_data.iterrows()
            ):

                target_col = col1 if index % 2 == 0 else col2

                with target_col:
                    st.metric(
                        label=f"Clase {row['Clase']}",
                        value=f"{row['Probabilidad']:.2f}%"
                    )

            st.bar_chart(
                probability_data.set_index("Clase")
            )

        # Información adicional del modelo
        model_info = resultado.get("model_info")

        if model_info:
            with st.expander("Información del modelo"):
                st.json(model_info)

        # Mostrar respuesta completa
        with st.expander("Ver respuesta JSON de la API"):
            st.json(resultado)

    except requests.exceptions.Timeout:

        st.error(
            "La API tardó demasiado en responder. "
            "Intenta nuevamente en unos segundos."
        )

    except requests.exceptions.HTTPError:

        st.error(
            f"Error HTTP {response.status_code}: "
            f"{response.text}"
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "No se pudo establecer conexión con FastAPI. "
            "Verifica que la API esté disponible."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"Error en la solicitud: {error}")

    except ValueError:

        st.error(
            "La API devolvió una respuesta que no contiene "
            "un JSON válido."
        )

    except Exception as error:

        st.error(f"Se produjo un error inesperado: {error}")


# ==========================================================
# 6. PIE DE PÁGINA
# ==========================================================

st.divider()

st.caption(
    "Proyecto 13MBID-ABR2627 | Machine Learning | "
    "Bank Marketing Prediction"
)
