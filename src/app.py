import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Spotify Genre Predictor",
    page_icon="🎵",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954 !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #1DB954 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }
    div[data-baseweb="slider"] div[role="slider"] + div,
    div[data-baseweb="popover"],
    div[data-baseweb="slider"] span {
        background-color: #282828 !important;
        color: #FFFFFF !important;
    }
    div[data-baseweb="slider"] p,
    div[data-baseweb="slider"] span {
        color: #FFFFFF !important;
    }
    div.stDownloadButton > button {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 10px 24px !important;
        width: 100%;
    }
    div.stDownloadButton > button:hover {
        background-color: #1ed760 !important;
        color: #000000 !important;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    ruta_modelo = "../models/modelo_final_ganador.pkl"
    if not os.path.exists(ruta_modelo):
        ruta_modelo = "models/modelo_final_ganador.pkl"
    return joblib.load(ruta_modelo)


clf = load_model()


def apply_feature_engineering(df_in):
    df_out = df_in.copy()
    loudness_abs = np.abs(df_out["loudness"]) + 1e-5
    df_out["energy_loudness_ratio"] = df_out["energy"] / loudness_abs
    df_out["acoustic_electric_ratio"] = df_out["acousticness"] / (
        df_out["energy"] + 1e-5
    )
    df_out["dance_energy_prod"] = df_out["danceability"] * df_out["energy"]
    df_out["mood_index"] = df_out["valence"] * df_out["danceability"]
    return df_out


# Título de la Aplicación
st.title("🎵 Clasificador de Géneros Musicales de Spotify")
st.write(
    "Ajusta los atributos en el panel lateral. El gráfico y la predicción se"
    " actualizarán en tiempo real."
)

# 4. BARRA LATERAL (Organizada en 2 COLUMNAS para los 10 atributos de ANOVA)
st.sidebar.header("🎛️ Atributos del Audio (Top 10 ANOVA)")

sb_col1, sb_col2 = st.sidebar.columns(2)

with sb_col1:
    acousticness = st.slider("Acústica", 0.0, 1.0, 0.20)
    energy = st.slider("Energía", 0.0, 1.0, 0.50)
    instrumentalness = st.slider("Instrumental", 0.0, 1.0, 0.00)
    loudness = st.slider("Sonoridad (dB)", -60.0, 0.0, -10.0)
    speechiness = st.slider("Voz hablada", 0.0, 1.0, 0.10)

with sb_col2:
    danceability = st.slider("Bailabilidad", 0.0, 1.0, 0.50)
    valence = st.slider("Positividad", 0.0, 1.0, 0.50)
    popularity = st.slider("Popularidad", 0, 100, 50)
    duration_min = st.slider("Duración (min)", 0.5, 10.0, 3.5)
    liveness = st.slider("En vivo", 0.0, 1.0, 0.10)

# 5. CONSTRUCCIÓN DE VARIABLES Y PREDICCIÓN AUTOMÁTICA
top_10_features = [
    "acousticness",
    "energy",
    "instrumentalness",
    "loudness",
    "speechiness",
    "danceability",
    "valence",
    "popularity",
    "duration_min",
    "liveness",
]

engineered_features = [
    "energy_loudness_ratio",
    "acoustic_electric_ratio",
    "dance_energy_prod",
    "mood_index",
]

all_features = top_10_features + engineered_features

raw_input_data = pd.DataFrame([
    {
        "acousticness": acousticness,
        "energy": energy,
        "instrumentalness": instrumentalness,
        "loudness": loudness,
        "speechiness": speechiness,
        "danceability": danceability,
        "valence": valence,
        "popularity": popularity,
        "duration_min": duration_min,
        "liveness": liveness,
    }
])

# Aplicar Feature Engineering y realizar la predicción en vivo
input_featured = apply_feature_engineering(raw_input_data)
X_input = input_featured[all_features]
prediccion_genero = clf.predict(X_input)[0]

# 6. ESTRUCTURA PRINCIPAL EN 2 COLUMNAS (Pantalla Dividida)
col_grafico, col_resultado = st.columns([1.2, 1])

# --- COLUMNA 1: Gráfico interactivo en vivo ---
with col_grafico:
    st.subheader("📊 Perfil de Atributos del Audio")

    # Preparar datos para el gráfico
    features_df = raw_input_data.T.reset_index()
    features_df.columns = ["Atributo", "Valor"]

    fig = px.bar(
        features_df,
        x="Atributo",
        y="Valor",
        labels={"Atributo": "Características", "Valor": "Nivel"},
        color="Valor",
        color_continuous_scale=["#052e16", "#1DB954", "#1ed760"],
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
    )

    st.plotly_chart(fig, use_container_width=True)

# --- COLUMNA 2: Resultado de Predicción y Datos ---
with col_resultado:
    st.subheader("🎧 Género Predicho")

   
    st.success(f"### 🎶 **{str(prediccion_genero).upper()}**")

    st.write("---")
    st.subheader(" Resumen del Perfil")

    df_pred = raw_input_data.copy()
    df_pred.insert(0, "genero_predicho", str(prediccion_genero).upper())

    st.dataframe(df_pred, use_container_width=True)

    # Botón de Descarga
    csv = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=" Descargar Pronóstico (CSV)",
        data=csv,
        file_name="pronostico_genero_spotify.csv",
        mime="text/csv",
    )

