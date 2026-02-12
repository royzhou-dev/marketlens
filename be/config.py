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

# Sentiment Analysis configuration
FINBERT_MODEL = os.getenv('FINBERT_MODEL', 'ProsusAI/finbert')
SENTIMENT_CACHE_TTL = int(os.getenv('SENTIMENT_CACHE_TTL', 15))  # minutes

# Reddit API (free - get credentials at reddit.com/prefs/apps)
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID', '')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET', '')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockAssistant/1.0')

# Twitter API (optional - requires paid tier $100+/month)
TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN', '')

# Agent configuration
AGENT_MAX_ITERATIONS = int(os.getenv('AGENT_MAX_ITERATIONS', 5))
