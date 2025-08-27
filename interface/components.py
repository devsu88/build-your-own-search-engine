"""
Componenti riutilizzabili per l'interfaccia Streamlit
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any
import requests

def method_comparison_widget(api_base_url: str):
    """
    Widget per confrontare diversi metodi di ricerca side-by-side
    """
    st.header("🔄 Confronto Metodi di Ricerca")
    
    # Input comune
    col1, col2 = st.columns(2)
    
    with col1:
        query = st.text_area(
            "Query di Ricerca",
            value="How to install Python?",
            height=80,
            help="Query da testare su tutti i metodi"
        )
        
        n_results = st.slider(
            "Numero Risultati",
            min_value=1,
            max_value=20,
            value=5
        )
    
    with col2:
        course_filter = st.selectbox(
            "Filtro Corso",
            options=["", "data-engineering-zoomcamp", "machine-learning-zoomcamp", "mlops-zoomcamp"],
            index=0
        )
        
        filters = {}
        if course_filter:
            filters["course"] = course_filter
    
    # Selezione metodi da confrontare
    st.subheader("📋 Metodi da Confrontare")
    methods_to_compare = st.multiselect(
        "Seleziona metodi",
        options=["tfidf", "svd", "nmf", "bert"],
        default=["tfidf", "svd", "nmf"],
        help="Scegli i metodi da confrontare"
    )
    
    if st.button("🔄 Confronta Metodi", type="primary"):
        if not query.strip():
            st.warning("⚠️ Inserisci una query di ricerca")
            return
        
        if not methods_to_compare:
            st.warning("⚠️ Seleziona almeno un metodo da confrontare")
            return
        
        # Esegue ricerche per tutti i metodi selezionati
        comparison_results = {}
        
        with st.spinner("🔄 Confrontando metodi..."):
            for method in methods_to_compare:
                try:
                    # Prepara i parametri specifici per ogni metodo
                    search_data = {
                        "query": query,
                        "method": method,
                        "n_results": n_results,
                        "boost": {"question": 2.0} if method == "tfidf" else {},
                        "filters": filters
                    }
                    
                    # Aggiungi parametri specifici per SVD/NMF
                    if method in ["svd", "nmf"]:
                        search_data["field"] = "text"
                        search_data["n_components"] = 16
                    elif method == "bert":
                        # BERT non ha bisogno di field o n_components
                        pass
                    else:
                        # TF-IDF usa field di default
                        search_data["field"] = "text"
                    
                    response = requests.post(f"{api_base_url}/search", json=search_data, timeout=30)
                    if response.status_code == 200:
                        comparison_results[method] = response.json()
                    else:
                        st.error(f"Errore per metodo {method}: {response.json().get('detail', 'Errore sconosciuto')}")
                        
                except Exception as e:
                    st.error(f"Errore per metodo {method}: {e}")
        
        if comparison_results:
            display_comparison_results(comparison_results, query)

def display_comparison_results(comparison_results: Dict[str, Any], query: str):
    """
    Mostra i risultati del confronto tra metodi
    """
    st.subheader(f"📊 Risultati Confronto: '{query}'")
    
    # Tabella comparativa
    comparison_data = []
    
    for method, results in comparison_results.items():
        if results and results['results']:
            for i, result in enumerate(results['results']):
                comparison_data.append({
                    'Metodo': method,
                    'Posizione': i + 1,
                    'Score': round(result['score'], 4),
                    'Corso': result['course'],
                    'Sezione': result['section'],
                    'Domanda': result['question'][:50] + "...",
                    'Tempo (s)': round(results['execution_time'], 3)
                })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch', hide_index=True)
        
        # Grafici comparativi
        col1, col2 = st.columns(2)
        
        with col1:
            # Grafico punteggi medi per metodo
            method_avg_scores = df_comparison.groupby('Metodo')['Score'].mean().reset_index()
            fig_scores = px.bar(
                method_avg_scores,
                x='Metodo',
                y='Score',
                title="Punteggio Medio per Metodo",
                color='Metodo'
            )
            st.plotly_chart(fig_scores, use_container_width=True)
        
        with col2:
            # Grafico tempi di esecuzione
            method_times = df_comparison.groupby('Metodo')['Tempo (s)'].mean().reset_index()
            fig_times = px.bar(
                method_times,
                x='Metodo',
                y='Tempo (s)',
                title="Tempo Medio di Esecuzione",
                color='Metodo'
            )
            st.plotly_chart(fig_times, use_container_width=True)
        
        # Statistiche dettagliate
        st.subheader("📈 Statistiche Dettagliate")
        
        for method, results in comparison_results.items():
            if results and results['results']:
                with st.expander(f"📊 {method.upper()}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risultati", len(results['results']))
                    
                    with col2:
                        st.metric("Tempo (s)", f"{results['execution_time']:.3f}")
                    
                    with col3:
                        scores = [r['score'] for r in results['results']]
                        st.metric("Score Medio", f"{sum(scores)/len(scores):.4f}")
                    
                    # Top 3 risultati
                    st.write("**Top 3 Risultati:**")
                    for i, result in enumerate(results['results'][:3]):
                        st.write(f"{i+1}. **{result['question'][:60]}...** (Score: {result['score']:.4f})")

def performance_analysis_widget(api_base_url: str):
    """
    Widget per analizzare le performance dei diversi metodi
    """
    st.header("⚡ Analisi Performance")
    
    # Query di test predefinite
    test_queries = [
        "How to install Python?",
        "What are the prerequisites?",
        "How to submit homework?",
        "When does the course start?",
        "Can I join late?"
    ]
    
    selected_queries = st.multiselect(
        "Query di Test",
        options=test_queries,
        default=test_queries[:3],
        help="Seleziona le query per testare le performance"
    )
    
    methods = ["tfidf", "svd", "nmf", "bert"]
    
    if st.button("🚀 Test Performance", type="primary"):
        if not selected_queries:
            st.warning("⚠️ Seleziona almeno una query di test")
            return
        
        # Esegue test di performance
        performance_data = []
        
        with st.spinner("🚀 Testando performance..."):
            for query in selected_queries:
                for method in methods:
                    try:
                        # Prepara i parametri specifici per ogni metodo
                        search_data = {
                            "query": query,
                            "method": method,
                            "n_results": 10,
                            "boost": {"question": 2.0} if method == "tfidf" else {},
                            "filters": {}
                        }
                        
                        # Aggiungi parametri specifici per SVD/NMF
                        if method in ["svd", "nmf"]:
                            search_data["field"] = "text"
                            search_data["n_components"] = 16
                        elif method == "bert":
                            # BERT non ha bisogno di field o n_components
                            pass
                        else:
                            # TF-IDF usa field di default
                            search_data["field"] = "text"
                        
                        response = requests.post(f"{api_base_url}/search", json=search_data, timeout=30)
                        if response.status_code == 200:
                            results = response.json()
                            performance_data.append({
                                'Query': query,
                                'Metodo': method,
                                'Tempo (s)': results['execution_time'],
                                'Risultati': results['total_results']
                            })
                        else:
                            # Metodo non disponibile (es. BERT)
                            performance_data.append({
                                'Query': query,
                                'Metodo': method,
                                'Tempo (s)': None,
                                'Risultati': 0
                            })
                            
                    except Exception as e:
                        performance_data.append({
                            'Query': query,
                            'Metodo': method,
                            'Tempo (s)': None,
                            'Risultati': 0
                        })
        
        if performance_data:
            display_performance_analysis(performance_data)

def display_performance_analysis(performance_data: List[Dict]):
    """
    Mostra l'analisi delle performance
    """
    df_perf = pd.DataFrame(performance_data)
    
    # Filtra solo i metodi disponibili
    available_methods = df_perf[df_perf['Tempo (s)'].notna()]['Metodo'].unique()
    
    if len(available_methods) == 0:
        st.warning("⚠️ Nessun metodo disponibile per l'analisi")
        return
    
    st.subheader("📊 Risultati Performance")
    
    # Tabella performance
    st.dataframe(df_perf, width='stretch', hide_index=True)
    
    # Grafici performance
    col1, col2 = st.columns(2)
    
    with col1:
        # Tempo medio per metodo
        method_avg_time = df_perf[df_perf['Tempo (s)'].notna()].groupby('Metodo')['Tempo (s)'].mean()
        fig_time = px.bar(
            x=method_avg_time.index,
            y=method_avg_time.values,
            title="Tempo Medio per Metodo",
            labels={'x': 'Metodo', 'y': 'Tempo (s)'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        # Numero risultati per metodo
        method_avg_results = df_perf[df_perf['Risultati'] > 0].groupby('Metodo')['Risultati'].mean()
        fig_results = px.bar(
            x=method_avg_results.index,
            y=method_avg_results.values,
            title="Risultati Medi per Metodo",
            labels={'x': 'Metodo', 'y': 'Numero Risultati'}
        )
        st.plotly_chart(fig_results, use_container_width=True)
    
    # Ranking performance
    st.subheader("🏆 Ranking Performance")
    
    # Calcola score complessivo (tempo + risultati)
    perf_summary = df_perf[df_perf['Tempo (s)'].notna()].groupby('Metodo').agg({
        'Tempo (s)': 'mean',
        'Risultati': 'mean'
    }).reset_index()
    
    # Normalizza i valori (0-1) e calcola score complessivo
    perf_summary['Tempo_Norm'] = 1 - (perf_summary['Tempo (s)'] / perf_summary['Tempo (s)'].max())
    perf_summary['Risultati_Norm'] = perf_summary['Risultati'] / perf_summary['Risultati'].max()
    perf_summary['Score_Complessivo'] = (perf_summary['Tempo_Norm'] + perf_summary['Risultati_Norm']) / 2
    
    # Ordina per score
    perf_summary = perf_summary.sort_values('Score_Complessivo', ascending=False)
    
    # Mostra ranking
    for i, row in perf_summary.iterrows():
        st.metric(
            f"#{i+1} {row['Metodo'].upper()}",
            f"Score: {row['Score_Complessivo']:.3f}",
            f"Tempo: {row['Tempo (s)']:.3f}s, Risultati: {row['Risultati']:.1f}"
        )
