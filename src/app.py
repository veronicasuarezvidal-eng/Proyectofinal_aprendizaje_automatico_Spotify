import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


clf = joblib.load('models/modelo_spotify_genre.pkl')

st.set_page_config(page_title="Spotify Genre Predictor", page_icon="🎵", layout="centered")

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

# Gráfico de barras interactivo con Plotly
st.subheader("📊 Perfil de Atributos del Audio")

# Transponer datos para graficar los atributos numéricos
features_df = input_data.drop(columns=['explicit']).T.reset_index()
features_df.columns = ['Atributo', 'Valor']

fig = px.bar(
    features_df,
    x='Atributo',
    y='Valor',
    title="Atributos del Audio Seleccionados",
    labels={'Atributo': 'Características', 'Valor': 'Nivel'},
    color='Valor',
    color_continuous_scale='Viridis'
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
##http://localhost:8501
##http://172.20.10.8:8501

