
import streamlit as st
import requests

# ==========================================
# CONFIGURACION
# ==========================================

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Predicción de clientes bancarios",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Predicción de suscripción bancaria")

st.write(
    "Completa la información del cliente para estimar "
    "la probabilidad de que suscriba un depósito a plazo fijo."
)

st.divider()

# ==========================================
# FORMULARIO
# ==========================================

with st.form("prediction_form"):

    st.subheader("1. Información personal")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Edad",
            min_value=17,
            max_value=100,
            value=35
        )

    with col2:
        job = st.selectbox(
            "Ocupación",
            [
                "admin.", "blue-collar", "entrepreneur",
                "housemaid", "management", "retired",
                "self-employed", "services", "student",
                "technician", "unemployed", "unknown"
            ]
        )

    with col3:
        marital = st.selectbox(
            "Estado civil",
            ["married", "single", "divorced", "unknown"]
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
            "¿Tiene crédito hipotecario?",
            ["yes", "no", "unknown"]
        )

    col1, col2 = st.columns(2)

    with col1:
        loan = st.selectbox(
            "¿Tiene préstamo personal?",
            ["yes", "no", "unknown"]
        )

    with col2:
        contact = st.selectbox(
            "Medio de contacto",
            ["cellular", "telephone"]
        )

    st.divider()

    st.subheader("2. Información de la campaña")

    col1, col2, col3 = st.columns(3)

    with col1:
        month = st.selectbox(
            "Mes de contacto",
            [
                "jan", "feb", "mar", "apr", "may",
                "jun", "jul", "aug", "sep", "oct",
                "nov", "dec"
            ]
        )

    with col2:
        day_of_week = st.selectbox(
            "Día de contacto",
            ["mon", "tue", "wed", "thu", "fri"]
        )

    with col3:
        duration = st.number_input(
            "Duración de llamada (segundos)",
            min_value=0,
            value=200
        )

    col1, col2 = st.columns(2)

    with col1:
        campaign = st.number_input(
            "Número de contactos en esta campaña",
            min_value=1,
            value=2
        )

    with col2:
        previous = st.number_input(
            "Contactos en campañas anteriores",
            min_value=0,
            value=0
        )

    col1, col2 = st.columns(2)

    with col1:
        pdays = st.number_input(
            "Días desde el último contacto",
            min_value=0,
            value=999,
            help="999 significa que el cliente no fue contactado anteriormente."
        )

    with col2:
        poutcome = st.selectbox(
            "Resultado de campaña anterior",
            ["nonexistent", "failure", "success"]
        )

    st.divider()

    st.subheader("3. Indicadores económicos")

    col1, col2, col3 = st.columns(3)

    with col1:
        emp_var_rate = st.number_input(
            "Variación de tasa de empleo",
            value=1.1
        )

    with col2:
        cons_price_idx = st.number_input(
            "Índice de precios al consumidor",
            value=93.994
        )

    with col3:
        cons_conf_idx = st.number_input(
            "Índice de confianza del consumidor",
            value=-36.4
        )

    col1, col2 = st.columns(2)

    with col1:
        euribor3m = st.number_input(
            "Euribor a 3 meses",
            value=4.857
        )

    with col2:
        nr_employed = st.number_input(
            "Número de empleados",
            value=5191.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Realizar predicción",
        use_container_width=True
    )


# ==========================================
# CONSUMO DE LA API
# ==========================================

if submitted:

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

        with st.spinner("Analizando información del cliente..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            st.success("Predicción realizada correctamente")

            st.subheader("Resultado de la clasificación")

            prediction = result["prediction"]
            probabilities = result["probability"]

            col1, col2 = st.columns(2)

            with col1:

                if prediction == "yes":
                    st.success("El modelo predice: SÍ")
                    st.write(
                        "El cliente tiene una predicción favorable "
                        "de suscripción."
                    )
                else:
                    st.warning("El modelo predice: NO")
                    st.write(
                        "El modelo no identifica una suscripción "
                        "favorable para este cliente."
                    )

            with col2:

                st.metric(
                    "Probabilidad de suscripción",
                    f"{probabilities.get('yes', 0) * 100:.2f}%"
                )

                st.metric(
                    "Probabilidad de no suscripción",
                    f"{probabilities.get('no', 0) * 100:.2f}%"
                )

            st.subheader("Distribución de probabilidades")

            st.progress(
                probabilities.get("yes", 0),
                text="Probabilidad de suscripción"
            )

            st.progress(
                probabilities.get("no", 0),
                text="Probabilidad de no suscripción"
            )

            st.caption(
                f"Modelo utilizado: {result['model_info']['model_type']}"
            )

        else:

            st.error(
                f"Error de la API ({response.status_code}): "
                f"{response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "No se pudo conectar con la API. "
            "Verifica que FastAPI esté funcionando."
        )

    except requests.exceptions.RequestException as e:

        st.error(f"Error en la comunicación con la API: {e}")


st.divider()

st.caption(
    "Proyecto de Machine Learning | Clasificación de clientes bancarios"
)
