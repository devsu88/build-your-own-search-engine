import requests
import pandas as pd
from typing import List, Dict, Any


def load_documents_from_github() -> List[Dict[str, Any]]:
    """
    Carica i documenti dal repository GitHub
    """
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    
    try:
        docs_response = requests.get(docs_url)
        docs_response.raise_for_status()
        documents_raw = docs_response.json()
        
        documents = []
        for course in documents_raw:
            course_name = course['course']
            for doc in course['documents']:
                doc['course'] = course_name
                documents.append(doc)
        
        return documents
        
    except requests.RequestException as e:
        print(f"Errore nel caricamento dei documenti: {e}")
        return []


def format_search_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Formatta i risultati della ricerca per una migliore visualizzazione
    """
    formatted_results = []
    
    for result in results:
        formatted_result = {
            'score': round(result.get('score', 0), 4),
            'course': result.get('course', ''),
            'section': result.get('section', ''),
            'question': result.get('question', ''),
            'text': result.get('text', '')[:200] + '...' if len(result.get('text', '')) > 200 else result.get('text', ''),
            'full_text': result.get('text', '')
        }
        formatted_results.append(formatted_result)
    
    return formatted_results


def get_search_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcola statistiche sui risultati della ricerca
    """
    if not results:
        return {}
    
    scores = [r.get('score', 0) for r in results]
    courses = [r.get('course', '') for r in results]
    sections = [r.get('section', '') for r in results]
    
    stats = {
        'total_results': len(results),
        'score_range': {
            'min': min(scores),
            'max': max(scores),
            'avg': sum(scores) / len(scores)
        },
        'courses_distribution': pd.Series(courses).value_counts().to_dict(),
        'sections_distribution': pd.Series(sections).value_counts().to_dict()
    }
    
    return stats

