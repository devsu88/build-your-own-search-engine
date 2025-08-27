import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD, NMF
from transformers import BertModel, BertTokenizer
from tqdm.auto import tqdm


class SearchEngine:
    """
    Motore di ricerca che implementa diversi metodi di ricerca semantica
    basato sul notebook tutorial
    """
    
    def __init__(self, text_fields=['section', 'question', 'text']):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}
        self.embeddings = None
        self.df = None
        self.bert_available = False
        
        # Inizializza BERT se disponibile
        try:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.model.eval()
            self.bert_available = True
            print("✅ BERT inizializzato correttamente")
        except Exception as e:
            print(f"⚠️  BERT non disponibile: {e}")
            self.bert_available = False
        
    def fit(self, records, vectorizer_params={'stop_words': 'english', 'min_df': 3}):
        """
        Prepara i vectorizer TF-IDF per tutti i campi di testo
        """
        self.df = pd.DataFrame(records)
        
        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv
    
    def search_tfidf(self, query, n_results=10, boost={}, filters={}):
        """
        Ricerca base usando TF-IDF + Cosine Similarity
        """
        if self.df is None:
            raise ValueError("Devi prima chiamare fit() per preparare il motore")
            
        score = np.zeros(len(self.df))
        
        # Calcola score per ogni campo con boost
        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s
        
        # Applica filtri
        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask
        
        # Ordina e restituisce risultati
        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx].copy()
        results['score'] = score[idx]
        
        return results.to_dict(orient='records')
    
    def search_svd(self, query, field='text', n_components=16, n_results=10, filters={}):
        """
        Ricerca usando SVD per riduzione dimensionalità
        """
        if self.df is None:
            raise ValueError("Devi prima chiamare fit() per preparare il motore")
            
        # Applica SVD
        svd = TruncatedSVD(n_components=n_components)
        X_emb = svd.fit_transform(self.matrices[field])
        
        # Trasforma la query
        q = self.vectorizers[field].transform([query])
        q_emb = svd.transform(q)
        
        # Calcola similarità
        score = cosine_similarity(X_emb, q_emb).flatten()
        
        # Applica filtri
        for field_name, value in filters.items():
            mask = (self.df[field_name] == value).values
            score = score * mask
        
        # Ordina e restituisce risultati
        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx].copy()
        results['score'] = score[idx]
        
        return results.to_dict(orient='records')
    
    def search_nmf(self, query, field='text', n_components=16, n_results=10, filters={}):
        """
        Ricerca usando NMF per matrix factorization
        """
        if self.df is None:
            raise ValueError("Devi prima chiamare fit() per preparare il motore")
            
        # Applica NMF
        nmf = NMF(n_components=n_components)
        X_emb = nmf.fit_transform(self.matrices[field])
        
        # Trasforma la query
        q = self.vectorizers[field].transform([query])
        q_emb = nmf.transform(q)
        
        # Calcola similarità
        score = cosine_similarity(X_emb, q_emb).flatten()
        
        # Ordina e restituisce risultati
        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx].copy()
        results['score'] = score[idx]
        
        return results.to_dict(orient='records')
    
    def load_bert_embeddings(self, filepath='embeddings.bin'):
        """
        Carica embeddings BERT pre-calcolati
        """
        try:
            with open(filepath, 'rb') as f:
                self.embeddings = pickle.load(f)
            print(f"✅ Embeddings BERT caricati da {filepath}")
            return True
        except FileNotFoundError:
            print(f"⚠️  File embeddings non trovato: {filepath}")
            return False
        except Exception as e:
            print(f"❌ Errore nel caricamento embeddings: {e}")
            return False
    
    def create_bert_embeddings(self, save_path='embeddings.bin'):
        """
        Crea embeddings BERT per tutti i documenti
        """
        if not self.bert_available:
            print("❌ BERT non disponibile. Installa torch e transformers")
            return False
            
        try:
            print("🤖 Creazione embeddings BERT...")
            print("   Questo processo richiede alcuni minuti...")
            
            # Crea embeddings per tutti i documenti
            embeddings = []
            for i, doc in enumerate(tqdm(self.df.to_dict('records'), desc="Creazione embeddings")):
                # Combina tutti i campi testuali
                text = f"{doc.get('text', '')} {doc.get('question', '')} {doc.get('section', '')}"
                
                # Tokenizza e crea embedding
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Usa l'ultimo hidden state e fai la media
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                    embeddings.append(embedding)
            
            # Salva embeddings
            self.embeddings = np.array(embeddings)
            with open(save_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            print(f"✅ Embeddings BERT creati e salvati in {save_path}")
            print(f"   📊 Shape: {self.embeddings.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Errore nella creazione embeddings: {e}")
            return False
    
    def load_or_create_bert_embeddings(self, filepath='embeddings.bin'):
        """
        Carica embeddings esistenti o li crea se mancanti
        """
        # Prima prova a caricare
        if self.load_bert_embeddings(filepath):
            return True
        
        # Se non esistono, crea automaticamente
        print("⚠️  Embeddings BERT non trovati")
        print("   🚀 Creazione automatica in corso...")
        return self.create_bert_embeddings(filepath)
    
    def search_bert(self, query, field='text', n_results=10, filters={}):
        """
        Ricerca usando embeddings BERT pre-calcolati
        Il parametro 'field' è ignorato per BERT che usa tutti i campi testuali
        """
        if self.embeddings is None:
            raise ValueError("BERT embeddings non caricati. Usa load_bert_embeddings()")
        
        if self.df is None:
            raise ValueError("Devi prima chiamare fit() per preparare il motore")
        
        # Calcola embedding della query
        with torch.no_grad():
            encoded_input = self.tokenizer([query], padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**encoded_input)
            query_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Calcola similarità
        score = cosine_similarity(self.embeddings, query_emb).flatten()
        
        # Applica filtri
        for field_name, value in filters.items():
            mask = (self.df[field_name] == value).values
            score = score * mask
        
        # Ordina e restituisce risultati
        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx].copy()
        results['score'] = score[idx]
        
        return results.to_dict(orient='records')
    
    def get_available_methods(self):
        """
        Restituisce i metodi di ricerca disponibili
        """
        methods = ['tfidf', 'svd', 'nmf']
        if self.embeddings is not None:
            methods.append('bert')
        return methods
    
    def get_available_courses(self):
        """
        Restituisce i corsi disponibili per i filtri
        """
        if self.df is not None:
            return self.df['course'].unique().tolist()
        return []
