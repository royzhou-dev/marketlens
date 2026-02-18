# CLAUDE.md

## Quick Start

```bash
cd be && pip install -r requirements.txt
# Add POLYGON_API_KEY and GEMINI_API_KEY to be/.env
# Optionally add REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, TWITTER_BEARER_TOKEN
python app.py
# Open http://localhost:5000
```

## Project Overview

Stock research assistant with Polygon.io data, AI chatbot (ReAct agent with tool calling), and social media sentiment analysis.

## Architecture

```
fe/                    Vanilla JS frontend (no frameworks)
  index.html             Main HTML structure
  app.js                 All frontend logic (~2400 lines)
  styles.css             Styling (dark/light theme support)
be/                    Python Flask backend
  app.py                 Main Flask app, stock data routes, blueprint registration
  config.py              Environment variables and feature settings
  agent_service.py       ReAct agent loop (tool calling + streaming)
  agent_tools.py         10 tool schemas + ToolExecutor with 3-layer cache
  llm_client.py          AgentLLMClient (google-genai SDK) + ConversationManager
  chat_routes.py         SSE streaming with structured events
  rag_pipeline.py        FAISS vector store, embeddings (google-genai SDK)
  scraper.py             ArticleScraper for extracting article content from URLs
  sentiment_service.py   Sentiment orchestrator (scraping + analysis + caching)
  sentiment_analyzer.py  FinBERT model wrapper (lazy-loaded)
  sentiment_routes.py    Sentiment API endpoints (Flask blueprint)
  social_scrapers.py     StockTwits, Reddit, Twitter scrapers
  forecast_service.py    LSTM forecasting service
  forecast_model.py      LSTM neural network model (PyTorch)
  forecast_routes.py     Forecast API endpoints (Flask blueprint)
  polygon_api.py         Polygon.io wrapper
  chat_service.py        Legacy RAG chat (replaced by agent_service.py)
  faiss_index/           Persistent FAISS vector store
  forecast_models/       Persisted LSTM models per ticker (model.pt, scaler.pkl, metadata.json)
```

## Chat Agent Architecture

The chat uses a **ReAct-style agent** with Gemini 2.0 Flash function calling:

1. User sends message → `agent_service.py` builds conversation contents
2. Gemini decides which tools to call based on the question
3. `ToolExecutor` runs tools with 3-layer caching (frontend context → server cache → API)
4. Results fed back to Gemini as function responses
5. Loop repeats (max 5 iterations) until Gemini returns a text response
6. Final text streamed to frontend via structured SSE events

**10 Available Tools:** `get_stock_quote`, `get_company_info`, `get_financials`, `get_news`, `search_knowledge_base`, `analyze_sentiment`, `get_price_forecast`, `get_dividends`, `get_stock_splits`, `get_price_history`

**SSE Event Protocol:**
- `event: tool_call` — Agent is calling a tool (status: calling/complete/error)
- `event: text` — Response text chunk
- `event: done` — Stream complete
- `event: error` — Fatal error

## Frontend

**7 Tabs:** Overview, Financials, Dividends, Splits, News, Sentiment, Forecast

- **Overview**: Price chart (line/candlestick, Canvas-based), key metrics, company description
- **Financials**: Income statement, balance sheet data
- **Dividends**: Dividend history table
- **Splits**: Stock split history table
- **News**: Latest articles, triggers RAG scraping for chat knowledge base
- **Sentiment**: Gauge visualization, filterable posts (bullish/neutral/bearish), engagement metrics
- **Forecast**: 30-day LSTM predictions table + chart overlay (Beta)

**Other Features:**
- Ticker selector with recent/popular tickers (localStorage)
- Dark/light theme
- Market status indicator
- Chat panel with SSE streaming + tool call status display
- Cache system with TTL tiers: STATIC (page refresh), DAILY (24h), MODERATE (30min), SHORT (15min)

## Key Technical Decisions

- **Frontend**: Pure vanilla JS - no frameworks allowed
- **AI SDK**: `google-genai` (new SDK) for chat + embeddings. Old `google-generativeai` still installed but only used by legacy `GeminiClient`
- **Agent**: Gemini function calling with manual dispatch (automatic calling disabled)
- **Agent model**: `gemini-2.0-flash` (configurable via GEMINI_MODEL env var)
- **Embeddings**: `gemini-embedding-001` (3072-dim, free)
- **Vector DB**: FAISS local storage (`be/faiss_index/`)
- **Scraping**: cloudscraper for Cloudflare bypass (StockTwits, Reddit)
- **Sentiment**: FinBERT model (`ProsusAI/finbert`, lazy-loaded, ~500MB)
- **Forecasting**: LSTM neural network (PyTorch), persisted per-ticker in `be/forecast_models/`
- **Graceful shutdown**: FAISS index saved via `atexit` handler in app.py

## API Keys (.env)

```bash
POLYGON_API_KEY=       # Stock data (5 calls/min free tier)
GEMINI_API_KEY=        # Chat + embeddings (free)
REDDIT_CLIENT_ID=      # Optional (free - reddit.com/prefs/apps)
REDDIT_CLIENT_SECRET=  # Optional
TWITTER_BEARER_TOKEN=  # Optional, paid ($100+/month)
```

## Rate Limits

- **Polygon.io**: 5 calls/min (free tier) — handled by 3-layer cache in ToolExecutor
  - Layer 1: Frontend sends cached data as `context` in chat request
  - Layer 2: Server-side TTL cache (5 min) in `ToolCache`
  - Layer 3: Live API call (last resort)
- **Gemini**: 15 RPM chat, 1500 RPM embeddings (free tier)

## Main Endpoints

### Stock Data (app.py)

| Endpoint | Purpose |
|----------|---------|
| `GET /` | Serve frontend |
| `GET /api/ticker/<ticker>/details` | Company info |
| `GET /api/ticker/<ticker>/previous-close` | Latest price data |
| `GET /api/ticker/<ticker>/aggregates` | Historical OHLCV (params: from, to, timespan) |
| `GET /api/ticker/<ticker>/news` | News articles (param: limit) |
| `GET /api/ticker/<ticker>/financials` | Financial statements |
| `GET /api/ticker/<ticker>/snapshot` | Current market snapshot |
| `GET /api/ticker/<ticker>/dividends` | Dividend history (param: limit) |
| `GET /api/ticker/<ticker>/splits` | Stock split history (param: limit) |
| `GET /api/market-status` | Market open/closed status |

### Chat Agent (chat_routes.py)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat/message` | Agent chat with structured SSE streaming |
| `POST /api/chat/scrape-articles` | RAG article indexing into FAISS |
| `GET /api/chat/conversations/<id>` | Get conversation history |
| `DELETE /api/chat/clear/<id>` | Clear conversation |
| `GET /api/chat/health` | Health check |
| `GET /api/chat/debug/chunks` | Debug FAISS vector store contents |

### Sentiment (sentiment_routes.py)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/sentiment/analyze` | Scrape + analyze ticker sentiment |
| `GET /api/sentiment/summary/<ticker>` | Get cached sentiment summary |
| `GET /api/sentiment/posts/<ticker>` | Get sentiment posts (filterable) |
| `GET /api/sentiment/health` | Health check |

### Forecast (forecast_routes.py)

| Endpoint | Purpose |
|----------|---------|
| `POST /api/forecast/predict/<ticker>` | Get LSTM price forecast |
| `POST /api/forecast/train/<ticker>` | Force train model |
| `GET /api/forecast/status/<ticker>` | Check model status |
| `GET /api/forecast/health` | Health check |

## Where to Find Details

| Topic | Location |
|-------|----------|
| Agent loop & tool calling | `be/agent_service.py`, `be/agent_tools.py` |
| Tool schemas (10 tools) | `be/agent_tools.py` - TOOL_DECLARATIONS |
| 3-layer caching | `be/agent_tools.py` - ToolExecutor._check_frontend_context() |
| LLM client (function calling) | `be/llm_client.py` - AgentLLMClient |
| Conversation history | `be/llm_client.py` - ConversationManager (24hr TTL, last 5 exchanges) |
| Configuration & env vars | `be/config.py` |
| Frontend SSE parser | `fe/app.js` - parseSSEBuffer(), sendChatMessage() |
| Frontend caching strategy | `fe/app.js` - stockCache object, CACHE_TTL constants |
| Chart implementation | `fe/app.js` - chartState, drawChart functions (Canvas-based) |
| RAG pipeline / FAISS | `be/rag_pipeline.py` |
| Article scraping for RAG | `be/scraper.py` - ArticleScraper |
| Sentiment orchestration | `be/sentiment_service.py` - SentimentService |
| Sentiment bias corrections | `be/sentiment_service.py` - aggregate calculation |
| FinBERT model wrapper | `be/sentiment_analyzer.py` - SentimentAnalyzer |
| Social scrapers | `be/social_scrapers.py` |
| Forecast service | `be/forecast_service.py` - ForecastService |
| LSTM model | `be/forecast_model.py` - ForecastModel (PyTorch) |
| Persisted models | `be/forecast_models/{ticker}/` (model.pt, scaler.pkl, metadata.json) |

## Development Rules

1. Keep frontend vanilla JS - no frameworks
2. Respect caching TTLs (see stockCache in app.js)
3. FAISS namespaces: `news:` for articles, `sentiment:` for posts
4. FinBERT has bullish bias - see sentiment_service.py for corrections
5. Tool responses use `role="tool"` in Gemini API (not "user")
6. Embeddings use `result.embeddings[0].values` with new google-genai SDK
7. `chat_service.py` is legacy — all new chat work goes through `agent_service.py`
