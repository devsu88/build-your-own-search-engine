#!/usr/bin/env python3
"""
Script di test per il backend del motore di ricerca
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from search_engine import SearchEngine
import requests
import time

def test_search_engine():
    """Testa la classe SearchEngine direttamente"""
    print("🧪 Testando la classe SearchEngine...")
    
    # Inizializza il motore
    engine = SearchEngine()
    
    # Carica i documenti dal repository GitHub
    print("  📥 Caricando documenti...")
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    
    documents = []
    for course in documents_raw:
        course_name = course['course']
        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    
    # Prepara il motore di ricerca
    engine.fit(documents)
    print(f"    ✅ Caricati {len(documents)} documenti")
    
    # Test con query di esempio
    query = "How to install Python?"
    
    try:
        # Test TF-IDF
        print("  📊 Testando TF-IDF...")
        results = engine.search_tfidf(
            query=query,
            n_results=3,
            boost={'question': 2.0},
            filters={'course': 'data-engineering-zoomcamp'}
        )
        print(f"    ✅ TF-IDF: {len(results)} risultati")
        
        # Test SVD
        print("  🔍 Testando SVD...")
        results = engine.search_svd(
            query=query,
            field='text',
            n_results=3,
            filters={'course': 'data-engineering-zoomcamp'}
        )
        print(f"    ✅ SVD: {len(results)} risultati")
        
        # Test NMF
        print("  📈 Testando NMF...")
        results = engine.search_nmf(
            query=query,
            field='text',
            n_results=3,
            filters={'course': 'data-engineering-zoomcamp'}
        )
        print(f"    ✅ NMF: {len(results)} risultati")
        
        print("  🎉 Tutti i test della classe SearchEngine sono passati!")
        
    except Exception as e:
        print(f"  ❌ Errore nei test: {e}")
        return False
    
    return True

def test_api():
    """Testa l'API FastAPI"""
    print("\n🌐 Testando l'API FastAPI...")
    
    base_url = "http://localhost:8000"
    
    try:
        # Test endpoint root
        print("  📍 Testando endpoint root...")
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Root: {data['message']}")
            print(f"    📋 Metodi disponibili: {data['available_methods']}")
        else:
            print(f"    ❌ Root: status {response.status_code}")
            return False
        
        # Test endpoint methods
        print("  🔧 Testando endpoint methods...")
        response = requests.get(f"{base_url}/methods")
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Methods: {len(data['methods'])} metodi disponibili")
        else:
            print(f"    ❌ Methods: status {response.status_code}")
            return False
        
        # Test endpoint search
        print("  🔍 Testando endpoint search...")
        search_data = {
            "query": "How to install Python?",
            "method": "tfidf",
            "n_results": 3,
            "boost": {"question": 2.0},
            "filters": {"course": "data-engineering-zoomcamp"}
        }
        
        response = requests.post(f"{base_url}/search", json=search_data)
        if response.status_code == 200:
            data = response.json()
            print(f"    ✅ Search: {data['total_results']} risultati in {data['execution_time']:.3f}s")
        else:
            print(f"    ❌ Search: status {response.status_code}")
            print(f"    📝 Response: {response.text}")
            return False
        
        print("  🎉 Tutti i test dell'API sono passati!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("  ❌ Impossibile connettersi all'API. Assicurati che sia in esecuzione.")
        return False
    except Exception as e:
        print(f"  ❌ Errore nei test API: {e}")
        return False

def main():
    """Funzione principale"""
    print("🚀 Test del Backend del Motore di Ricerca")
    print("=" * 50)
    
    # Test della classe SearchEngine
    if not test_search_engine():
        print("\n❌ Test della classe SearchEngine falliti!")
        return
    
    # Test dell'API (se è in esecuzione)
    if not test_api():
        print("\n⚠️  Test dell'API falliti. Avvia l'API con 'python app/main.py'")
        print("   Poi esegui nuovamente questo script.")
        return
    
    print("\n🎉 Tutti i test sono passati! Il backend è pronto.")

if __name__ == "__main__":
    main()
