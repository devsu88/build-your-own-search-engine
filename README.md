# Build Your Own Search Engine

Tutorial completo per costruire un motore di ricerca personalizzato con diversi algoritmi di ricerca semantica.

## 🚀 Caratteristiche

- **Backend API** completo con FastAPI
- **Frontend interattivo** con Streamlit
- **4 metodi di ricerca** implementati
- **Confronto side-by-side** dei metodi
- **Analisi performance** automatica
- **Interfaccia intuitiva** per test e confronti

## 🏗️ Struttura del Progetto

```
build-your-own-search-engine/
├── app/                    # Backend API
│   ├── __init__.py
│   ├── main.py            # API FastAPI principale
│   ├── search_engine.py   # Classe motore di ricerca
│   ├── config.py          # Configurazione
│   └── utils.py           # Funzioni di utilità
├── interface/              # Frontend Streamlit
│   ├── streamlit_app.py   # Interfaccia ricerca base
│   ├── components.py      # Componenti riutilizzabili
│   └── main_app.py        # App principale con navigazione
├── notebook.ipynb         # Tutorial originale
├── embeddings.bin         # Embeddings BERT pre-calcolati
└── requirements.txt       # Dipendenze Python
```

## 🔧 Metodi di Ricerca Implementati

1. **TF-IDF + Cosine Similarity**: Metodo base con vectorizzazione TF-IDF
2. **SVD**: Riduzione dimensionalità con TruncatedSVD
3. **NMF**: Matrix factorization non-negativa
4. **BERT**: Ricerca semantica con embeddings pre-calcolati

## 📦 Installazione e Avvio

### 1. Installare le dipendenze
```bash
pip install -r requirements.txt
```

### 2. Avviare il backend
```bash
cd app
python main.py
```

L'API sarà disponibile su `http://localhost:8000`

### 3. Avviare il frontend
```bash
cd interface
streamlit run main_app.py
```

L'interfaccia sarà disponibile su `http://localhost:8501`

## 🌐 Utilizzo

### Backend API
- **Documentazione Swagger**: `http://localhost:8000/docs`
- **Health check**: `http://localhost:8000/health`
- **Endpoint ricerca**: `POST /search`

### Frontend Streamlit
- **🔍 Ricerca Base**: Testa singoli metodi di ricerca
- **🔄 Confronto Metodi**: Confronta side-by-side diversi metodi
- **⚡ Analisi Performance**: Analizza performance e ranking
- **📚 Informazioni**: Documentazione e troubleshooting

## 💡 Esempi di Utilizzo

### Ricerca Base
```python
import requests

response = requests.post("http://localhost:8000/search", json={
    "query": "How to install Python?",
    "method": "tfidf",
    "n_results": 5,
    "boost": {"question": 2.0},
    "filters": {"course": "data-engineering-zoomcamp"}
})

results = response.json()
print(f"Trovati {results['total_results']} risultati in {results['execution_time']:.3f}s")
```

### Confronto Metodi
1. Apri l'interfaccia Streamlit
2. Vai alla pagina "🔄 Confronto Metodi"
3. Inserisci una query
4. Seleziona i metodi da confrontare
5. Analizza i risultati side-by-side

### Test Performance
1. Vai alla pagina "⚡ Analisi Performance"
2. Seleziona le query di test
3. Esegui il test automatico
4. Visualizza il ranking delle performance

## 🔍 Funzionalità Avanzate

### Boost Factors (TF-IDF)
- **Question Boost**: Peso maggiore per domande
- **Section Boost**: Peso per sezioni
- **Text Boost**: Peso per testo completo

### Filtri
- **Corso**: Filtra per corso specifico
- **Sezione**: Filtra per sezione
- **Estendibile**: Facile aggiungere nuovi filtri

### Visualizzazioni
- **Grafici punteggi**: Bar chart orizzontali
- **Distribuzione corsi**: Grafici a torta
- **Confronto metodi**: Grafici comparativi
- **Ranking performance**: Metriche e classifiche

## 🛠️ Configurazione

Modifica `app/config.py` per personalizzare:
- Parametri dei vectorizer
- Numero di componenti per SVD/NMF
- Limiti sui risultati
- Configurazione API

## 🧪 Testing

### Test Backend
```bash
python test_backend.py
```

### Test Frontend
```bash
cd interface
streamlit run main_app.py
```

## 📊 Metriche e Performance

- **Tempo di esecuzione** per ogni ricerca
- **Punteggi di similarità** per ogni risultato
- **Confronto performance** tra metodi
- **Ranking automatico** basato su tempo e qualità

## 🔧 Troubleshooting

### API non raggiungibile
- Verifica che il backend sia in esecuzione
- Controlla la porta 8000
- Riavvia il backend se necessario

### Metodo BERT non disponibile
- Gli embeddings BERT sono opzionali
- L'app funziona senza BERT
- Altri metodi rimangono disponibili

### Ricerca lenta
- Riduci il numero di risultati
- Usa filtri per restringere la ricerca
- SVD/NMF sono più veloci di BERT

## 🚀 Sviluppi Futuri

- [ ] Supporto per nuovi metodi di ricerca
- [ ] Cache intelligente per risultati frequenti
- [ ] Export risultati in vari formati
- [ ] Dashboard analytics avanzate
- [ ] Supporto per dataset personalizzati