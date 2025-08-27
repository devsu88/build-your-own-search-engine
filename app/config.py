"""
Configurazione per l'API del motore di ricerca
"""

# Configurazione API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Search Engine API"
API_VERSION = "1.0.0"

# Configurazione motore di ricerca
DEFAULT_TEXT_FIELDS = ['section', 'question', 'text']
DEFAULT_VECTORIZER_PARAMS = {
    'stop_words': 'english',
    'min_df': 3
}

# Configurazione metodi di ricerca
SEARCH_METHODS = {
    'tfidf': {
        'name': 'TF-IDF + Cosine Similarity',
        'description': 'Metodo base con vectorizzazione TF-IDF e similarità del coseno',
        'supports_boost': True,
        'supports_filters': True
    },
    'svd': {
        'name': 'SVD + Dimensionality Reduction',
        'description': 'Riduzione dimensionalità con TruncatedSVD',
        'supports_boost': False,
        'supports_filters': True
    },
    'nmf': {
        'name': 'NMF + Matrix Factorization',
        'description': 'Matrix factorization non-negativa',
        'supports_boost': False,
        'supports_filters': True
    },
    'bert': {
        'name': 'BERT + Semantic Search',
        'description': 'Ricerca semantica con embeddings BERT pre-calcolati',
        'supports_boost': False,
        'supports_filters': True
    }
}

# Configurazione embeddings
EMBEDDINGS_FILE = 'embeddings.bin'
BERT_MODEL_NAME = 'bert-base-uncased'
SVD_N_COMPONENTS = 16
NMF_N_COMPONENTS = 16

# Configurazione risultati
DEFAULT_N_RESULTS = 10
MAX_N_RESULTS = 100

