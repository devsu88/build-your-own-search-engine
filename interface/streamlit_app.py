#!/usr/bin/env python3
"""
Frontend Streamlit per il motore di ricerca
Interfaccia interattiva per testare diversi metodi di ricerca semantica
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Configurazione pagina
st.set_page_config(
    page_title="Search Engine Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurazione API
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Verifica se l'API è disponibile"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_api_info():
    """Ottiene informazioni dall'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.json()
    except:
        return None

def search_documents(query, method, n_results, boost, filters, field, n_components):
    """Esegue una ricerca tramite API"""
    try:
        search_data = {
            "query": query,
            "method": method,
            "n_results": n_results,
            "boost": boost,
            "filters": filters,
            "field": field,
            "n_components": n_components
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=search_data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Errore API: {response.json().get('detail', 'Errore sconosciuto')}")
            return None
    except Exception as e:
        st.error(f"Errore di connessione: {e}")
        return None

def create_score_chart(results):
    """Crea un grafico dei punteggi"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    df = df.sort_values('score', ascending=True)
    
    fig = px.bar(
        df, 
        x='score', 
        y=range(len(df)),
        orientation='h',
        title="Punteggi di Similarità",
        labels={'score': 'Score', 'y': 'Risultato'}
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Punteggio Similarità",
        yaxis_title="Risultato"
    )
    
    return fig

def create_course_distribution_chart(results):
    """Crea un grafico della distribuzione per corso"""
    if not results:
        return None
    
    df = pd.DataFrame(results)
    course_counts = df['course'].value_counts()
    
    fig = px.pie(
        values=course_counts.values,
        names=course_counts.index,
        title="Distribuzione Risultati per Corso"
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    """Funzione principale dell'app"""
    
    # Header (rimosso per evitare duplicazione)
    # st.title("🔍 Search Engine Tester")
    # st.markdown("Interfaccia interattiva per testare diversi metodi di ricerca semantica")
    
    # Sidebar per configurazione
    with st.sidebar:
        st.header("⚙️ Configurazione Ricerca")
        
        # Query input
        query = st.text_area(
            "Query di Ricerca",
            value="How to install Python?",
            height=100,
            help="Inserisci la tua domanda o query di ricerca"
        )
        
        # Metodo di ricerca
        method = st.selectbox(
            "Metodo di Ricerca",
            options=["tfidf", "svd", "nmf", "bert"],
            index=0,
            help="Scegli il metodo di ricerca da utilizzare"
        )
        
        # Numero risultati
        n_results = st.slider(
            "Numero Risultati",
            min_value=1,
            max_value=50,
            value=10,
            help="Numero di risultati da restituire"
        )
        
        # Campo di ricerca (per SVD/NMF)
        if method in ["svd", "nmf"]:
            field = st.selectbox(
                "Campo di Ricerca",
                options=["text", "question", "section"],
                index=0,
                help="Campo su cui eseguire la ricerca (per SVD/NMF)"
            )
        elif method == "bert":
            field = "text"  # BERT usa tutti i campi
            st.info("ℹ️ BERT usa automaticamente tutti i campi testuali combinati")
        else:
            field = "text"
        
        # Componenti (per SVD/NMF)
        n_components = st.slider(
            "Numero Componenti",
            min_value=2,
            max_value=50,
            value=16,
            help="Numero di componenti per SVD/NMF"
        )
        
        # Boost factors (solo per TF-IDF)
        if method == "tfidf":
            st.subheader("🎯 Boost Factors")
            question_boost = st.slider("Question Boost", 0.0, 5.0, 1.0, 0.1)
            section_boost = st.slider("Section Boost", 0.0, 5.0, 1.0, 0.1)
            text_boost = st.slider("Text Boost", 0.0, 5.0, 1.0, 0.1)
            
            boost = {
                "question": question_boost,
                "section": section_boost,
                "text": text_boost
            }
        else:
            boost = {}
        
        # Filtri
        st.subheader("🔍 Filtri")
        course_filter = st.selectbox(
            "Corso",
            options=["", "data-engineering-zoomcamp", "machine-learning-zoomcamp", "mlops-zoomcamp"],
            index=0,
            help="Filtra per corso specifico"
        )
        
        filters = {}
        if course_filter:
            filters["course"] = course_filter
        
        # Pulsante ricerca
        search_button = st.button("🔍 Esegui Ricerca", type="primary", use_container_width=True)
    
    # Contenuto principale
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Risultati Ricerca")
        
        if search_button:
            if not query.strip():
                st.warning("⚠️ Inserisci una query di ricerca")
            else:
                with st.spinner("🔍 Ricerca in corso..."):
                    results = search_documents(
                        query=query,
                        method=method,
                        n_results=n_results,
                        boost=boost,
                        filters=filters,
                        field=field,
                        n_components=n_components
                    )
                
                if results:
                    # Mostra statistiche
                    st.success(f"✅ Trovati {results['total_results']} risultati in {results['execution_time']:.3f} secondi")
                    
                    # Mostra risultati in tabella
                    if results['results']:
                        df_results = pd.DataFrame(results['results'])
                        
                        # Formatta i risultati per la visualizzazione
                        display_df = df_results[['score', 'course', 'section', 'question', 'text']].copy()
                        display_df['score'] = display_df['score'].round(4)
                        display_df['text'] = display_df['text'].str[:100] + "..."
                        
                        st.dataframe(
                            display_df,
                            width='stretch',
                            hide_index=True
                        )
                        
                        # Mostra testo completo per il primo risultato
                        if len(results['results']) > 0:
                            with st.expander("📖 Testo Completo Primo Risultato"):
                                first_result = results['results'][0]
                                st.write(f"**Domanda:** {first_result['question']}")
                                st.write(f"**Sezione:** {first_result['section']}")
                                st.write(f"**Corso:** {first_result['course']}")
                                st.write(f"**Score:** {first_result['score']:.4f}")
                                st.write("**Testo:**")
                                st.write(first_result['text'])
                    else:
                        st.info("ℹ️ Nessun risultato trovato")
    
    with col2:
        st.subheader("📈 Visualizzazioni")
        
        if search_button and query.strip():
            results = search_documents(
                query=query,
                method=method,
                n_results=n_results,
                boost=boost,
                filters=filters,
                field=field,
                n_components=n_components
            )
            
            if results and results['results']:
                # Grafico punteggi
                score_fig = create_score_chart(results['results'])
                if score_fig:
                    st.plotly_chart(score_fig, use_container_width=True)
                
                # Grafico distribuzione corsi
                course_fig = create_course_distribution_chart(results['results'])
                if course_fig:
                    st.plotly_chart(course_fig, use_container_width=True)
    
    # Footer con informazioni API
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if check_api_health():
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")
    
    with col2:
        api_info = get_api_info()
        if api_info:
            st.info(f"📊 {api_info.get('total_results', 'N/A')} documenti")
    
    with col3:
        # Box orario rimosso
        pass

if __name__ == "__main__":
    main()
