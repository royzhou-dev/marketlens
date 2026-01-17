import os
from dotenv import load_dotenv

load_dotenv()

# Existing
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
PORT = int(os.getenv('PORT', 5000))

# Gemini configuration (for chat and embeddings - FREE)
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')

# FAISS configuration (path relative to this file's location)
FAISS_INDEX_PATH = os.getenv('FAISS_INDEX_PATH',
    os.path.join(os.path.dirname(__file__), 'faiss_index'))

# Embedding settings (using Google's free embedding model)
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')
MAX_CONTEXT_LENGTH = int(os.getenv('MAX_CONTEXT_LENGTH', 8000))
RAG_TOP_K = int(os.getenv('RAG_TOP_K', 5))
