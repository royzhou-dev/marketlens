# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

```bash
# 1. Install dependencies
cd be && pip install -r requirements.txt

# 2. Configure .env (add your Gemini API key)
# Get free key at: https://aistudio.google.com/apikey

# 3. Run the server
python app.py

# 4. Open http://localhost:5000
```

## Project Overview

Project A is a personal stock trading assistant that provides users with comprehensive market information powered by the Polygon.io API. The platform features an AI-powered chatbot assistant with RAG (Retrieval Augmented Generation) capabilities that can answer questions about stocks using real-time data, financial statements, and scraped news articles. Users can explore stock data, news, financials, dividends, splits, and receive intelligent analysis from the AI assistant.

## Architecture

- **fe/**: Vanilla JavaScript frontend (no frameworks)
  - Single-page application with tab-based navigation
  - Client-side caching system with configurable TTL
  - Canvas-based chart rendering (line chart + candlestick views)
  - Sliding chat sidebar UI for AI assistant
  - Responsive design with mobile support

- **be/**: Python Flask backend
  - RESTful API endpoints
  - Polygon.io API integration
  - AI chatbot with RAG pipeline (scraping, embeddings, vector search)
  - Google Gemini integration for intelligent responses (FREE)
  - FAISS vector database for local semantic search
  - CORS-enabled for local development
  - Environment-based configuration

## Tech Stack

### Frontend
- Vanilla JavaScript (ES6+)
- HTML5 Canvas for charts
- CSS3 (Flexbox & Grid)
- No external dependencies

### Backend
- Flask 3.0.0
- Python 3.8+
- Google Gemini API (gemini-2.0-flash for chat, text-embedding-004 for embeddings - FREE)
- FAISS vector database (local file-based)
- NumPy for vector operations
- BeautifulSoup4 + lxml for web scraping
- tenacity for retry logic
- Requests for API calls
- python-dotenv for configuration
- Flask-CORS for cross-origin support

## Key Features

### Core Stock Data
1. **Stock Selection**: Searchable dropdown with thousands of tickers from `company_tickers.json`
2. **Market Status**: Real-time market open/closed indicator
3. **Dark Mode**: Theme toggle with system preference detection, persisted to localStorage
4. **Overview Tab**: Key metrics (price, volume, market cap, open, high, low) + interactive price chart with Line/Candlestick toggle
5. **Financials Tab**: Quarterly/annual financial statements (revenue, net income, assets, liabilities)
6. **Dividends Tab**: Historical dividend payment data with ex-dates and pay dates
7. **Splits Tab**: Stock split history with ratios and types
8. **News Tab**: Latest news articles related to selected stocks

### AI Chatbot Assistant
9. **Intelligent Q&A**: Ask natural language questions about stocks and receive data-driven answers
10. **Context-Aware**: Chatbot has access to all stock data visible in the frontend (overview, financials, news, etc.)
11. **RAG Pipeline**: Full news article scraping and semantic search using FAISS local vector database
12. **Streaming Responses**: Real-time LLM output with smooth user experience
13. **Conversation History**: Maintains context within chat sessions
14. **Automatic Scraping**: Articles are scraped and indexed automatically when a stock is selected
15. **Sliding Sidebar UI**: Non-intrusive chat interface that slides in from the right

## Data Flow

### Stock Data Flow
1. User selects a ticker from dropdown
2. Frontend makes parallel API calls to backend for ticker details and previous close
3. Backend fetches data from Polygon.io API
4. Data is cached client-side with appropriate TTL
5. User switches tabs → data loads from cache if available, otherwise fetches from API
6. Subsequent visits to same ticker/tab load instantly from cache

### AI Chat Flow
1. User opens chat sidebar and asks a question
2. Frontend collects current stock context (ticker, cached data from all tabs)
3. POST to `/api/chat/message` with context payload
4. Backend (ChatService) orchestrates:
   - Generate query embedding using Google Gemini
   - Search FAISS local index for relevant article contexts
   - Assemble comprehensive prompt with stock data + RAG contexts + conversation history
   - Stream Gemini response back to frontend
5. Frontend displays streaming response in chat UI

### RAG Data Sources
The chatbot receives context from two different sources:

| Data Type | Storage | How it reaches LLM |
|-----------|---------|-------------------|
| News articles | FAISS vector DB (`be/faiss_index/`) | RAG semantic search on `full_content` |
| Overview/Price | Frontend cache (browser memory) | Passed directly in each request |
| Financials | Frontend cache (browser memory) | Passed if query contains financial keywords |
| Dividends | Frontend cache (browser memory) | Passed if query mentions "dividend" |
| Splits | Frontend cache (browser memory) | Passed if query mentions "split" |

**News articles** are the only data embedded into vectors. When a stock is selected:
1. `preloadNewsForRAG()` fetches news metadata from Polygon.io
2. `scrapeAndEmbedArticles()` sends articles to backend for scraping
3. Backend scrapes full article content, generates embeddings, and stores in FAISS
4. Metadata (title, source, URL, `full_content`) saved to `be/faiss_index/metadata.json`

**Financial data** (overview, financials, dividends, splits) is ephemeral - fetched from Polygon.io, cached in browser memory, and sent fresh with each chat message.

## Caching Strategy

The application uses client-side caching to respect Polygon.io's free tier rate limits (5 API calls/minute):

- **Static data** (cache until page refresh): Ticker details, dividends, splits
- **Daily data** (24-hour cache): Previous close, market status, chart data
- **Moderate** (30-minute cache): Financial statements
- **Short** (15-minute cache): News articles

Access cache stats via browser console: `stockCache.getStats()`

## API Integration

### Polygon.io API
All stock data fetched from Polygon.io API (https://polygon.io/):
- `/v3/reference/tickers/{ticker}` - Company details
- `/v2/aggs/ticker/{ticker}/prev` - Previous day close
- `/v2/aggs/ticker/{ticker}/range/...` - Historical aggregates (OHLCV data for charts)
- `/v3/reference/dividends` - Dividend history
- `/v3/reference/splits` - Stock split history
- `/v2/reference/news` - News articles (metadata)
- `/vX/reference/financials` - Financial statements
- `/v1/marketstatus/now` - Market status

Note: Related companies endpoint (`/v1/related-companies/{ticker}`) was removed due to limited free tier data.

### Google Gemini API
AI chatbot and embeddings (https://aistudio.google.com/) - **100% FREE**:
- Chat Model: gemini-2.0-flash (FREE tier)
- Embedding Model: text-embedding-004 (768 dimensions, FREE)
- `generate_content` - Streaming chat responses
- `embed_content` - Vector embeddings for RAG
- System instructions for stock analyst persona

### FAISS Vector Database
Local file-based vector database for semantic search:
- Index Type: IndexFlatIP (768 dimensions, cosine similarity)
- Storage: `be/faiss_index/` directory
- Persistence: Manual save on batch operations and graceful shutdown
- Metadata: Stored separately in JSON files alongside vector index

## API Rate Limits

| API | Limit | Notes |
|-----|-------|-------|
| Polygon.io (free) | 5 calls/min | Handled by frontend caching |
| Gemini Chat | 15 RPM, 1M tokens/day | FREE tier |
| Gemini Embeddings | 1500 RPM | FREE tier |
| FAISS | Unlimited | Local storage |

## Configuration

API keys and settings stored in `.env` file:
```bash
# Stock data API
POLYGON_API_KEY=your_polygon_key_here
PORT=5000

# Google Gemini (for chat and embeddings - FREE)
GEMINI_API_KEY=your_gemini_key_here

# FAISS Vector Database (optional)
FAISS_INDEX_PATH=be/faiss_index

# Optional chatbot configurations
GEMINI_MODEL=gemini-2.0-flash
EMBEDDING_MODEL=text-embedding-004
MAX_CONTEXT_LENGTH=8000
RAG_TOP_K=5
```

## Backend Module Structure

### Core Modules
- **app.py**: Main Flask application with stock data routes
- **polygon_api.py**: Polygon.io API wrapper class
- **config.py**: Environment configuration loader

### AI Chatbot Modules
- **chat_routes.py**: Flask routes for chat endpoints
- **chat_service.py**: Orchestration layer coordinating RAG pipeline and LLM
- **scraper.py**: Web scraper for extracting full article content (BeautifulSoup)
- **rag_pipeline.py**: Embedding generation and FAISS vector store management
- **llm_client.py**: Google Gemini client with streaming and conversation management

## Chat API Endpoints

- `POST /api/chat/message` - Main chat endpoint (streaming responses)
  - Request: `{ticker, message, context: {overview, financials, news, ...}, conversation_id}`
  - Response: Server-Sent Events stream of LLM chunks

- `POST /api/chat/scrape-articles` - Background article scraping job
  - Request: `{ticker, articles: [...]}`
  - Response: `{scraped: N, embedded: N, failed: N, skipped: N}`

- `GET /api/chat/conversations/<id>` - Get conversation history
- `DELETE /api/chat/clear/<id>` - Clear conversation
- `GET /api/chat/health` - Health check for chat service
- `GET /api/chat/debug/chunks` - Debug endpoint to view stored RAG chunks
  - Query params: `ticker` (optional filter), `limit` (default 50)
  - Returns: `{total, returned, index_stats, chunks: [{id, doc_id, ticker, title, source, full_content, ...}]}`

## Important Implementation Notes

### AI Chatbot Architecture
- **Vanilla JS Maintained**: Chat UI built with vanilla JavaScript following existing patterns (no frameworks)
- **RAG Strategy**: Automatic scraping when stock is selected, embeddings saved to local FAISS index
- **Shared VectorStore**: `ChatService` and `ContextRetriever` share the same `VectorStore` instance to ensure consistency
- **Context Assembly**: Chatbot receives full stock context from frontend (all cached data) + RAG results from FAISS
- **Streaming**: Server-Sent Events for real-time LLM response streaming
- **Persistence**: FAISS index saved automatically after batch operations and on graceful shutdown
- **Error Handling**: Graceful fallback to article descriptions if scraping fails; traceback logging for debugging
- **Cost Optimization**: Uses 100% free Google Gemini APIs for chat and embeddings, FAISS is free and local, client-side caching reduces API calls

### Price Chart Implementation
- **Dual View Modes**: Toggle between line chart and candlestick chart (no additional API calls - OHLC data already fetched)
- **chartState Object**: Centralized state for canvas, data, hover index, animation, and view mode
- **High DPI Support**: Canvas scales for Retina/4K displays
- **Interactive Features**: Crosshair tracking, tooltips (OHLC in candle mode), animated drawing
- **Theme-Aware**: `getChartColors()` returns appropriate colors for light/dark mode
- **Time Ranges**: 1M, 3M, 6M, 1Y, 5Y with 24-hour caching per range

### Code Organization Principles
- Backend modules are fully independent and can be used separately
- Frontend chat code is isolated in dedicated section of app.js
- All existing functionality remains unchanged (additive implementation)
- Chat styling isolated in dedicated CSS section
- No breaking changes to existing API endpoints
- Design system documented in `.design-engineer/system.md`

### Development Guidelines
- **No Frameworks**: Keep frontend pure vanilla JavaScript
- **Maintain Caching**: Respect client-side caching strategy for all API calls
- **Error Transparency**: Log scraping/embedding failures but continue operation
- **User Privacy**: Conversations stored in-memory only (cleared after 24 hours)
- **Zero Cost**: All APIs (Gemini chat + embeddings) are free, FAISS is local
- **Local Storage**: FAISS index stored on disk, no cloud database costs

## Future Enhancements

- ~~AI-powered stock analysis and recommendations~~ ✅ **IMPLEMENTED**
- ~~Dark mode~~ ✅ **IMPLEMENTED**
- Recently viewed stocks (localStorage persistence)
- Portfolio tracking with AI insights
- Real-time price updates (WebSocket integration)
- Advanced technical indicators with AI interpretation
- User authentication and preferences
- Watchlists and alerts with AI notifications
- Multi-ticker comparison in chat
- Voice input for chat queries
- Export chat conversations
- Fine-tuned model on historical stock analysis