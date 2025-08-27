import time
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn

from search_engine import SearchEngine

# Inizializza FastAPI
app = FastAPI(
    title="Search Engine API",
    description="API per testare diversi metodi di ricerca semantica",
    version="1.0.0"
)

# CORS per permettere richieste dal frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelli Pydantic per le richieste
class SearchRequest(BaseModel):
    query: str
    method: str = "tfidf"
    n_results: int = 10
    boost: Dict[str, float] = {}
    filters: Dict[str, str] = {}
    field: str = "text"
    n_components: int = 16

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    method: str
    query: str
    execution_time: float
    total_results: int

# Inizializza il motore di ricerca
search_engine = SearchEngine()

@app.on_event("startup")
async def startup_event():
    """Inizializza il motore di ricerca all'avvio"""
    try:
        print("🚀 Inizializzazione motore di ricerca...")
        
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
        print("  🔧 Preparando motore di ricerca...")
        search_engine.fit(documents)
        
        # Controllo BERT embeddings
        print("  🤖 Controllando disponibilità BERT...")
        bert_available = search_engine.load_or_create_bert_embeddings('embeddings.bin')
        
        if bert_available:
            print("  ✅ BERT disponibile - metodo 'bert' attivo")
        else:
            print("  ⚠️  BERT non disponibile - metodo 'bert' disabilitato")
        
        print(f"🎉 Motore di ricerca inizializzato con {len(documents)} documenti")
        
    except Exception as e:
        print(f"❌ Errore durante l'inizializzazione: {e}")
        print("⚠️  L'app si avvia senza alcune funzionalità")

@app.get("/")
async def root():
    """Endpoint root"""
    return {
        "message": "Search Engine API",
        "version": "1.0.0",
        "available_methods": search_engine.get_available_methods(),
        "available_courses": search_engine.get_available_courses()
    }

@app.get("/methods")
async def get_methods():
    """Restituisce i metodi di ricerca disponibili"""
    return {
        "methods": search_engine.get_available_methods(),
        "courses": search_engine.get_available_courses()
    }

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Esegue una ricerca usando il metodo specificato"""
    
    start_time = time.time()
    
    # Validazione input base
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La query non può essere vuota")
    
    if request.n_results <= 0 or request.n_results > 100:
        raise HTTPException(status_code=400, detail="n_results deve essere tra 1 e 100")
    
    if request.method in ["svd", "nmf"] and (request.n_components <= 0 or request.n_components > 100):
        raise HTTPException(status_code=400, detail="n_components deve essere tra 1 e 100")
    
    try:
        # Esegue la ricerca in base al metodo
        if request.method == "tfidf":
            results = search_engine.search_tfidf(
                query=request.query,
                n_results=request.n_results,
                boost=request.boost,
                filters=request.filters
            )
        elif request.method == "svd":
            results = search_engine.search_svd(
                query=request.query,
                field=request.field,
                n_components=request.n_components,
                n_results=request.n_results,
                filters=request.filters
            )
        elif request.method == "nmf":
            results = search_engine.search_nmf(
                query=request.query,
                field=request.field,
                n_components=request.n_components,
                n_results=request.n_results,
                filters=request.filters
            )
        elif request.method == "bert":
            # Controlla se BERT è disponibile
            if "bert" not in search_engine.get_available_methods():
                raise HTTPException(
                    status_code=400, 
                    detail="Metodo BERT non disponibile. Gli embeddings non sono stati caricati correttamente."
                )
            
            results = search_engine.search_bert(
                query=request.query,
                field=request.field,
                n_results=request.n_results,
                filters=request.filters
            )
        else:
            raise HTTPException(status_code=400, detail=f"Metodo non supportato: {request.method}")
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            results=results,
            method=request.method,
            query=request.query,
            execution_time=execution_time,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore durante la ricerca: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "engine_ready": search_engine.df is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

