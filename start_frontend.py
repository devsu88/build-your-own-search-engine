#!/usr/bin/env python3
"""
Script di avvio per il frontend del motore di ricerca
"""

import subprocess
import sys
import os

def main():
    """Avvia il frontend Streamlit"""
    
    print("🚀 Avvio Frontend Search Engine")
    print("=" * 40)
    
    # Verifica che siamo nella directory corretta
    if not os.path.exists("interface/main_app.py"):
        print("❌ Errore: File main_app.py non trovato")
        print("   Assicurati di essere nella directory root del progetto")
        return
    
    # Verifica che Streamlit sia installato
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} installato")
    except ImportError:
        print("❌ Streamlit non installato")
        print("   Installa con: pip install streamlit")
        return
    
    print("\n📱 Avvio interfaccia Streamlit...")
    print("   L'interfaccia sarà disponibile su: http://localhost:8501")
    print("   Premi Ctrl+C per fermare")
    print("-" * 40)
    
    try:
        # Avvia Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "interface/main_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Frontend fermato dall'utente")
    except Exception as e:
        print(f"\n❌ Errore nell'avvio: {e}")

if __name__ == "__main__":
    main()
