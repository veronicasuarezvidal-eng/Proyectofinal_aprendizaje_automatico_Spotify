import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Spotify Genre Predictor", page_icon="🎵", layout="centered") 

# Inyección de CSS corregida para un look tipo Spotify impecable
st.markdown("""
    <style>
    /* 1. Fondo general y sidebar */
    .stApp {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }

    /* 2. Títulos y etiquetas principales */
    h1, h2, h3, h4, h5, h6 {
        color: #1DB954 !important;
    }
    [data-testid="stWidgetLabel"] p {
        color: #1DB954 !important;
        font-weight: 600;
    }

    /* 3. Corrección de los Sliders */
    
    /* Pista activa (línea de la izquierda) y bolita del slider */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #1DB954 !important;
        border-color: #1DB954 !important;
    }
    
    /* Fondo del recuadro del número (Tooltip / Value) */
    div[data-baseweb="slider"] div[role="slider"] + div,
    div[data-baseweb="popover"],
    div[data-baseweb="slider"] span {
        background-color: #282828 !important;
        color: #FFFFFF !important;
    }

    /* Números de min, max y valor seleccionado */
    div[data-baseweb="slider"] p, 
    div[data-baseweb="slider"] span {
        color: #FFFFFF !important;
    }

    /* 4. Botones estilo Spotify (Verde con texto negro y bordes redondeados) */
    div.stButton > button, div.stDownloadButton > button {
        background-color: #1DB954 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: all 0,2s ease;
    }
    
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #1ed760 !important;
        color: #000000 !important;
        transform: scale(1,03);
    }
    </style>
""", unsafe_allow_html=True)

clf = joblib.load('models/modelo_spotify_genre.pkl')

st.title("🎵 Clasificador de Géneros Musicales de Spotify")
st.write("Ajusta las características del audio en el panel izquierdo para predecir el género musical.")

st.sidebar.header("🎛️ Atributos del Audio")

danceability = st.sidebar.slider("Bailabilidad (Danceability)", 0.0, 1.0, 0.50)
energy = st.sidebar.slider("Energía (Energy)", 0.0, 1.0, 0.50)
loudness = st.sidebar.slider("Sonoridad (Loudness dB)", -60.0, 0.0, -10.0)
speechiness = st.sidebar.slider("Voz hablada (Speechiness)", 0.0, 1.0, 0.10)
acousticness = st.sidebar.slider("Acústica (Acousticness)", 0.0, 1.0, 0.20)
instrumentalness = st.sidebar.slider("Instrumentalidad (Instrumentalness)", 0.0, 1.0, 0.00)
liveness = st.sidebar.slider("En vivo (Liveness)", 0.0, 1.0, 0.10)
valence = st.sidebar.slider("Positividad (Valence)", 0.0, 1.0, 0.50)
tempo = st.sidebar.slider("Tempo (BPM)", 50.0, 220.0, 120.0)
popularity = st.sidebar.slider("Popularidad (Popularity)", 0, 100, 50)
explicit = st.sidebar.selectbox("¿Contenido Explícito?", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")


input_data = pd.DataFrame([{
    'danceability': danceability,
    'energy': energy,
    'loudness': loudness,
    'speechiness': speechiness,
    'acousticness': acousticness,
    'instrumentalness': instrumentalness,
    'liveness': liveness,
    'valence': valence,
    'tempo': tempo,
    'popularity': popularity,
    'explicit': explicit
}])


st.subheader("📊 Perfil de Atributos del Audio")
features_df = input_data.drop(columns=['explicit']).T.reset_index()
features_df.columns = ['Atributo', 'Valor']

fig = px.bar(
    features_df,
    x='Atributo',
    y='Valor',
    title="Atributos del Audio Seleccionados",
    labels={'Atributo': 'Características', 'Valor': 'Nivel'},
    color='Valor',
    color_continuous_scale=['#052e16', '#1DB954', '#1ed760']
)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color="#FFFFFF")
)

st.plotly_chart(fig, use_container_width=True)


if st.button("🔮 Predecir Género"):
    prediction = clf.predict(input_data)[0]
    
  
    st.markdown("## 🎶 🎵 🎶 🎵 🎶 🎵 🎶 🎵")
    st.success(f"🎧 El género predicho para este perfil musical es: **{str(prediction).upper()}**")

    df_pred = input_data.copy()
    df_pred.insert(0, 'genero_predicho', str(prediction).upper())
    
    st.subheader('Tabla Detallada')
    st.dataframe(df_pred, use_container_width=True)
    
    csv = df_pred.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label='Descargar Pronóstico en CSV',
        data=csv,
        file_name='pronostico_genero_spotify.csv',
        mime='text/csv'
    )




