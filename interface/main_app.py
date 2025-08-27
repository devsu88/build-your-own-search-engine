#!/usr/bin/env python3
"""
App Streamlit principale per il motore di ricerca
Interfaccia completa con navigazione tra funzionalità
"""

import streamlit as st
from streamlit_app import main as search_interface
from components import method_comparison_widget, performance_analysis_widget
import requests

# Configurazione pagina
st.set_page_config(
    page_title="Search Engine Tester",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurazione API
API_BASE_URL = "http://localhost:8000"

def check_api_status():
    """Verifica lo stato dell'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            return False, None
    except:
        return False, None

def main():
    """Funzione principale dell'app"""
    
    # Header principale
    st.title("🔍 Search Engine Tester")
    st.markdown("Interfaccia completa per testare e confrontare diversi metodi di ricerca semantica")
    
    # Controllo stato API
    api_online, api_data = check_api_status()
    
    if not api_online:
        st.error("🔴 **API non disponibile**")
        st.warning("""
        Per utilizzare questa interfaccia, avvia prima il backend:
        
        ```bash
        cd app
        python main.py
        ```
        
        L'API sarà disponibile su http://localhost:8000
        """)
        return
    
    # Sidebar per navigazione
    with st.sidebar:
        st.success("🟢 **API Online**")
        
        # Informazioni API
        try:
            api_info = requests.get(f"{API_BASE_URL}/").json()
            # Conta i documenti dal dataset
            if 'available_courses' in api_info:
                st.info(f"📊 **Dataset caricato** con corsi disponibili")
            st.info(f"🔧 **{len(api_info.get('available_methods', []))}** metodi disponibili")
        except:
            pass
        
        st.markdown("---")
        
        # Navigazione
        st.header("🧭 Navigazione")
        page = st.selectbox(
            "Seleziona Pagina",
            options=[
                "🔍 Ricerca Base",
                "🔄 Confronto Metodi", 
                "⚡ Analisi Performance",
                "📚 Informazioni"
            ],
            index=0
        )
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Azioni Rapide")
        if st.button("🔄 Ricarica API", use_container_width=True):
            st.rerun()
    
    # Contenuto principale in base alla pagina selezionata
    if page == "🔍 Ricerca Base":
        st.header("🔍 Ricerca Base")
        st.markdown("Interfaccia per testare singoli metodi di ricerca")
        search_interface()
        
    elif page == "🔄 Confronto Metodi":
        st.header("🔄 Confronto Metodi")
        st.markdown("Confronta side-by-side diversi metodi di ricerca")
        method_comparison_widget(API_BASE_URL)
        
    elif page == "⚡ Analisi Performance":
        st.header("⚡ Analisi Performance")
        st.markdown("Analizza le performance dei diversi metodi")
        performance_analysis_widget(API_BASE_URL)
        
    elif page == "📚 Informazioni":
        st.header("📚 Informazioni e Documentazione")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Metodi Disponibili")
            
            try:
                methods_info = requests.get(f"{API_BASE_URL}/methods").json()
                
                for method in methods_info.get('methods', []):
                    if method == 'tfidf':
                        st.info(f"**{method.upper()}** - TF-IDF + Cosine Similarity")
                        st.write("Supporta boost factors e filtri avanzati")
                    elif method == 'svd':
                        st.info(f"**{method.upper()}** - SVD + Dimensionality Reduction")
                        st.write("Riduce dimensionalità mantenendo informazioni principali")
                    elif method == 'nmf':
                        st.info(f"**{method.upper()}** - NMF + Matrix Factorization")
                        st.write("Matrix factorization non-negativa")
                    elif method == 'bert':
                        st.info(f"**{method.upper()}** - BERT + Semantic Search")
                        st.write("Ricerca semantica con embeddings pre-calcolati")
                        st.success("✅ Disponibile")
                    else:
                        st.info(f"**{method.upper()}** - Metodo non riconosciuto")
                        st.warning("⚠️  Stato sconosciuto")
                    
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Errore nel caricamento metodi: {e}")
        
        with col2:
            st.subheader("📊 Statistiche Dataset")
            
            try:
                # Prova a ottenere informazioni sui corsi
                methods_info = requests.get(f"{API_BASE_URL}/methods").json()
                courses = methods_info.get('courses', [])
                
                if courses:
                    st.write(f"**Corsi disponibili:** {len(courses)}")
                    for course in courses:
                        st.write(f"• {course}")
                    
                    # Informazioni aggiuntive sul dataset
                    st.write("")
                    st.write("**Informazioni:**")
                    st.write("• Dataset caricato correttamente")
                    st.write("• 948 documenti disponibili")
                    st.write("• Metodi di ricerca attivi")
                else:
                    st.write("Nessun corso disponibile")
                    
            except Exception as e:
                st.error(f"Errore nel caricamento statistiche: {e}")
        
        # Esempi di utilizzo
        st.subheader("💡 Esempi di Utilizzo")
        
        st.markdown("""
        ### 🔍 Ricerca Base
        - **Query:** "How to install Python?"
        - **Metodo:** tfidf
        - **Boost:** question=2.0
        - **Filtri:** course="data-engineering-zoomcamp"
        
        ### 🔄 Confronto Metodi
        - **Query:** "Course prerequisites"
        - **Metodi:** tfidf, svd, nmf
        - **Risultati:** 5 per metodo
        - **Analisi:** Punteggi e tempi a confronto
        
        ### ⚡ Test Performance
        - **Query multiple:** Query predefinite
        - **Metodi:** Tutti disponibili
        - **Metriche:** Tempo, risultati, score complessivo
        """)
        
        # Gestione BERT
        st.subheader("🤖 Gestione Embeddings BERT")
        
        st.info("""
        **Come funziona BERT in questa app:**
        
        • **Primo avvio**: Se non esistono embeddings, l'app li crea automaticamente
        • **Tempo creazione**: Circa 5-10 minuti per 948 documenti
        • **Salvataggio**: Gli embeddings vengono salvati in `embeddings.bin`
        • **Riavvii successivi**: Caricamento istantaneo degli embeddings esistenti
        
        **Se BERT non è disponibile:**
        • Verifica che `torch` e `transformers` siano installati
        • Controlla la connessione internet per il download del modello
        • L'app funziona comunque con altri metodi (TF-IDF, SVD, NMF)
        """)
        
        # Troubleshooting
        st.subheader("🔧 Troubleshooting")
        
        with st.expander("Problemi Comuni"):
            st.markdown("""
            **❌ API non raggiungibile**
            - Verifica che il backend sia in esecuzione
            - Controlla la porta 8000
            - Riavvia il backend se necessario
            
            **❌ Metodo BERT non disponibile**
            - Gli embeddings BERT sono opzionali
            - L'app funziona senza BERT
            - Altri metodi rimangono disponibili
            - **Soluzione**: Riavvia il backend per creazione automatica
            
            **❌ Ricerca lenta**
            - Riduci il numero di risultati
            - Usa filtri per restringere la ricerca
            - SVD/NMF sono più veloci di BERT
            """)

if __name__ == "__main__":
    main()
