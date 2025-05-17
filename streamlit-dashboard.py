# Dashboard létrehozása Streamlit és Plotly segítségével

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import re

# Beállítások
st.set_page_config(page_title='Kibertámadási Adatok Elemzése', 
                  page_icon=':lock:', layout='wide')

# Adatok betöltése
@st.cache_data
def load_data():
    # A második sort használjuk fejlécként
    df = pd.read_excel(r"c:\Users\User\Downloads\midonazest.xlsx", header=1)
    
    # Dátumformátum konvertálása
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        # Év, hónap kinyerése
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month
        df['quarter'] = df['event_date'].dt.quarter
    
    return df

# Dashboard cím
st.title('Kibertámadási Adatok Elemzése és Vizualizációja')
st.markdown('Ez a dashboard a kibertámadási adatok elemzésére szolgál, visual analitikai eszközökkel.')

# Adatok betöltése
try:
    df = load_data()
    st.success(f"Sikeresen betöltve {df.shape[0]} esemény adata")
except Exception as e:
    st.error(f"Hiba az adatok betöltése során: {e}")
    st.stop()

# Oldalsáv - Szűrők
st.sidebar.title('Szűrők')

# Év szűrő
years = sorted(df['year'].unique())
selected_years = st.sidebar.multiselect('Évek', years, default=years[-5:])

# Actor típus szűrő
actor_types = sorted(df['actor_type'].unique())
selected_actor_types = st.sidebar.multiselect('Actor típusok', actor_types, default=actor_types)

# Iparág szűrő
top_industries = df['industry'].value_counts().head(10).index.tolist()
selected_industries = st.sidebar.multiselect('Iparágak (Top 10)', top_industries, default=top_industries[:5])

# Ország szűrő
top_countries = df['country'].value_counts().head(15).index.tolist()
selected_countries = st.sidebar.multiselect('Célországok (Top 15)', top_countries, default=['United States of America'])

# Adatok szűrése
filtered_df = df[
    (df['year'].isin(selected_years)) &
    (df['actor_type'].isin(selected_actor_types))
]

# Ha selected_industries vagy selected_countries nem üres, akkor szűrjünk ezekre is
if selected_industries:
    filtered_df = filtered_df[filtered_df['industry'].isin(selected_industries)]
if selected_countries:
    filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

# Metrikák: KPI-k
st.header('Kulcsmutatók')
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_events = len(filtered_df)
    st.metric("Összes esemény", f"{total_events:,}")

with col2:
    most_common_actor = filtered_df['actor_type'].value_counts().idxmax()
    st.metric("Leggyakoribb támadó típus", most_common_actor)

with col3:
    most_targeted_industry = filtered_df['industry'].value_counts().idxmax()
    st.metric("Legtöbbet támadott iparág", most_targeted_industry)

with col4:
    most_common_motive = filtered_df['motive'].value_counts().idxmax()
    st.metric("Leggyakoribb motívum", most_common_motive)

# Időbeli trendek - Interaktív Plotly Line Chart
st.header('Időbeli trendek')

# Időbeli aggregálás: évente vagy negyedévente
time_agg = st.radio(
    "Időbeli aggregálás", 
    ('Évente', 'Negyedévente'),
    horizontal=True
)

if time_agg == 'Évente':
    time_series = filtered_df.groupby('year').size().reset_index(name='count')
    
    fig = px.line(time_series, x='year', y='count', 
                 title='Kibertámadások száma évente',
                 labels={'year': 'Év', 'count': 'Események száma'},
                 markers=True)
    
    # Y tengely kezdete 0-nál
    fig.update_layout(yaxis_range=[0, time_series['count'].max() * 1.1])
    
    st.plotly_chart(fig, use_container_width=True)
else:
    # Negyedévenkénti elemzés
    filtered_df['year_quarter'] = filtered_df['year'].astype(str) + '-Q' + filtered_df['quarter'].astype(str)
    time_series_q = filtered_df.groupby('year_quarter').size().reset_index(name='count')
    
    fig = px.line(time_series_q, x='year_quarter', y='count', 
                 title='Kibertámadások száma negyedévente',
                 labels={'year_quarter': 'Év-Negyedév', 'count': 'Események száma'},
                 markers=True)
    
    # X tengely címkék elforgatása
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, time_series_q['count'].max() * 1.1]
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Események típus szerinti bontása évente
yearly_event_types = filtered_df.groupby(['year', 'event_type']).size().reset_index(name='count')
yearly_event_types = yearly_event_types.sort_values('year')

fig = px.bar(yearly_event_types, x='year', y='count', color='event_type',
             title='Eseménytípusok évente',
             labels={'year': 'Év', 'count': 'Események száma', 'event_type': 'Esemény típus'},
             barmode='stack')

st.plotly_chart(fig, use_container_width=True)

# Actor típusok és iparágak vizualizációja
st.header('Actor típusok és célpontok elemzése')
col1, col2 = st.columns(2)

with col1:
    # Actor típusok Pie Chart
    actor_counts = filtered_df['actor_type'].value_counts()
    fig = px.pie(values=actor_counts.values, names=actor_counts.index,
                title='Actor típusok eloszlása',
                hole=0.4)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Top 10 iparág Bar Chart
    industry_counts = filtered_df['industry'].value_counts().head(10)
    fig = px.bar(x=industry_counts.index, y=industry_counts.values,
                title='Top 10 célpont iparág',
                labels={'x': 'Iparág', 'y': 'Események száma'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Hőtérkép: Actor típusok és motívumok kereszttáblája
st.header('Actor típusok és motívumok összefüggései')

# Kereszttábla készítése
heatmap_data = pd.crosstab(filtered_df['actor_type'], filtered_df['motive'])

# Csak a leggyakoribb motívumok megjelenítése a jobb láthatóság érdekében
top_motives = filtered_df['motive'].value_counts().head(8).index
heatmap_data = heatmap_data[top_motives]

# Plotly Heatmap
fig = px.imshow(heatmap_data.values,
               x=heatmap_data.columns,
               y=heatmap_data.index,
               labels=dict(x="Motívum", y="Actor típus", color="Események száma"),
               title="Actor típusok és motívumok összefüggései")

fig.update_layout(
    xaxis_tickangle=-45,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Földrajzi eloszlás - Choropleth térkép
st.header('Kibertámadások földrajzi eloszlása')

# Országok szerinti események száma
country_events = filtered_df['country'].value_counts().reset_index()
country_events.columns = ['country', 'events']

# Világtérkép a támadások száma szerint
fig = px.choropleth(country_events, 
                   locations='country', 
                   locationmode='country names',
                   color='events',
                   hover_name='country',
                   color_continuous_scale=px.colors.sequential.Plasma,
                   title='Kibertámadások száma országonként')

# Térkép méretezése és formázása
fig.update_layout(
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth'
    )
)

st.plotly_chart(fig, use_container_width=True)

# Ország-ország hálózati grafikon (egyszerűsített)
st.header('Támadó és célországok kapcsolata')

# Csak a significant country párokat választjuk ki, ahol legalább N támadás történt
country_pairs = filtered_df.groupby(['actor_country', 'country']).size().reset_index(name='count')
min_attacks = st.slider('Minimum támadások száma a megjelenítéshez', 1, 50, 10)
significant_pairs = country_pairs[country_pairs['count'] >= min_attacks]

if not significant_pairs.empty:
    # Jelentős ország párok megjelenítése
    fig = px.scatter(significant_pairs, x='actor_country', y='country', size='count', color='count',
                    hover_data=['count'],
                    title=f'Támadó és célországok kapcsolata (min. {min_attacks} támadás)',
                    labels={'actor_country': 'Támadó ország', 'country': 'Célország', 'count': 'Támadások száma'})
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(f"Nincs ország pár, ahol a támadások száma legalább {min_attacks} lenne a jelenlegi szűrőkkel.")

# Szöveges elemzés - WordCloud a leírásokból
st.header('Kibertámadások leírásainak szöveges elemzése')

try:
    # NLTK adatok letöltése (egyszer kell végrehajtani)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Leírások tisztítása
    descriptions = filtered_df['description'].dropna().astype(str)
    
    # Stop szavak beállítása
    stop_words = set(stopwords.words('english'))
    additional_stop_words = {'the', 'and', 'to', 'of', 'in', 'a', 'for', 'on', 'with', 'by', 'from', 'that', 'was', 'were', 'been'}
    stop_words.update(additional_stop_words)
    
    # Szöveg tokenizálása és tisztítása
    all_words = []
    for desc in descriptions:
        words = word_tokenize(desc.lower())
        filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    
    # WordCloud generálása
    if all_words:
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                             max_words=100, contour_width=3, contour_color='steelblue').generate(' '.join(all_words))
        
        # Matplotlib ábrán megjelenítés
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.info("Nincs elegendő szöveges adat a szófelhő generálásához.")
        
    # Top 20 szó gyakorisága
    from collections import Counter
    word_freq = Counter(all_words).most_common(20)
    
    if word_freq:
        word_df = pd.DataFrame(word_freq, columns=['Szó', 'Gyakoriság'])
        
        fig = px.bar(word_df, x='Szó', y='Gyakoriság',
                    title='Top 20 leggyakoribb szó a támadások leírásában',
                    color='Gyakoriság')
        
        st.plotly_chart(fig, use_container_width=True)
    
except Exception as e:
    st.error(f"Hiba a szöveges elemzés során: {e}")

# Predikciós komponens
st.header('Új kibertámadás esemény típusának predikciója')
st.write("Tölts ki néhány adatot egy hipotetikus kibertámadásról a típusának előrejelzéséhez.")

try:
    # Modellek betöltése
    rf_model = joblib.load('random_forest_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Predikciós űrlap
    col1, col2 = st.columns(2)
    
    with col1:
        pred_actor_type = st.selectbox('Támadó típusa', sorted(df['actor_type'].unique()))
        pred_industry = st.selectbox('Célpont iparág', sorted(df['industry'].unique()))
        pred_motive = st.selectbox('Motívum', sorted(df['motive'].unique()))
        
    with col2:
        pred_country = st.selectbox('Célpont ország', sorted(df['country'].unique()))
        pred_actor_country = st.selectbox('Támadó ország', sorted(df['actor_country'].unique()))
        pred_year = st.number_input('Év', min_value=2023, max_value=2025, value=2024)
        pred_month = st.number_input('Hónap', min_value=1, max_value=12, value=5)
    
    pred_description = st.text_area('Támadás leírása', 
                                  'Ransomware attack targeting a financial institution with data encryption and theft.')
    
    # Predict gomb
    if st.button('Esemény típus előrejelzése'):
        # Adatok előkészítése a predikcióhoz
        example_data = {
            'actor_type': pred_actor_type,
            'industry': pred_industry,
            'motive': pred_motive,
            'country': pred_country,
            'actor_country': pred_actor_country,
            'year': pred_year,
            'month': pred_month,
            'description': pred_description
        }
        
        # Import a predikciós függvény
        from predict_module import predict_event_type
        
        # Predikció készítése
        try:
            prediction, probabilities = predict_event_type(
                example_data, rf_model, tfidf_vectorizer, label_encoders, scaler
            )
            
            # Eredmények megjelenítése
            st.success(f"A predikált esemény típus: **{prediction}**")
            
            # Valószínűségek vizualizációja
            sorted_probas = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            proba_df = pd.DataFrame(sorted_probas, columns=['Esemény típus', 'Valószínűség'])
            
            fig = px.bar(proba_df, x='Esemény típus', y='Valószínűség',
                        title='Esemény típusok valószínűsége',
                        color='Valószínűség')
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Hiba a predikció során: {e}")
            st.info("Győződj meg róla, hogy a predict_module.py fájl elérhető a megfelelő predikciós függvénnyel.")
    
except Exception as e:
    st.warning(f"A predikciós modell nem található: {e}")
    st.info("A predikciós funkció használatához először futtasd le a gépi tanulási szkriptet a modellek létrehozásához.")

# Footer
st.markdown("---")
st.markdown("Kibertámadási Adatok Elemző Dashboard | Készítette: [Az Ön Neve] | Adatforrás: UMSPP adathalmaz")
