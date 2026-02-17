# CLAUDE.md

## Quick Start

```bash
cd be && pip install -r requirements.txt
# Add POLYGON_API_KEY and GEMINI_API_KEY to .env
python app.py
# Open http://localhost:5000
```

## Project Overview

Stock research assistant with Polygon.io data, AI chatbot (ReAct agent with tool calling), and social media sentiment analysis.

## Architecture

```
fe/          Vanilla JS frontend (no frameworks)
be/          Python Flask backend
  app.py              Main Flask app, stock data routes
  agent_service.py    ReAct agent loop (tool calling + streaming)
  agent_tools.py      10 tool schemas + ToolExecutor with 3-layer cache
  llm_client.py       AgentLLMClient (google-genai SDK) + ConversationManager
  chat_routes.py      SSE streaming with structured events
  rag_pipeline.py     FAISS vector store, embeddings (google-genai SDK)
  sentiment_*.py      Social sentiment (FinBERT, scrapers)
  forecast_*.py       LSTM price forecasting
  polygon_api.py      Polygon.io wrapper
  chat_service.py     Legacy RAG chat (replaced by agent_service.py)
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

## Key Technical Decisions

- **Frontend**: Pure vanilla JS - no frameworks allowed
- **AI SDK**: `google-genai` (new SDK) for chat + embeddings. Old `google-generativeai` still installed but only used by legacy `GeminiClient`
- **Agent**: Gemini function calling with manual dispatch (automatic calling disabled)
- **Vector DB**: FAISS local storage (`be/faiss_index/`)
- **Scraping**: cloudscraper for Cloudflare bypass (StockTwits, Reddit)
- **Sentiment**: FinBERT model (lazy-loaded, ~500MB)

## API Keys (.env)

```bash
POLYGON_API_KEY=     # Stock data (5 calls/min free tier)
GEMINI_API_KEY=      # Chat + embeddings (free)
TWITTER_BEARER_TOKEN= # Optional, paid ($100+/month)
```

## Rate Limits

- **Polygon.io**: 5 calls/min (free tier) — handled by 3-layer cache in ToolExecutor
  - Layer 1: Frontend sends cached data as `context` in chat request
  - Layer 2: Server-side TTL cache (5 min) in `ToolCache`
  - Layer 3: Live API call (last resort)
- **Gemini**: 15 RPM chat, 1500 RPM embeddings (free tier)

## Main Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /api/chat/message` | Agent chat with structured SSE streaming |
| `POST /api/chat/scrape-articles` | RAG article indexing into FAISS |
| `POST /api/sentiment/analyze` | Sentiment analysis |
| `POST /api/forecast/predict/<ticker>` | LSTM price forecast |

## Where to Find Details

| Topic | Location |
|-------|----------|
| Agent loop & tool calling | `be/agent_service.py`, `be/agent_tools.py` |
| Tool schemas (10 tools) | `be/agent_tools.py` - TOOL_DECLARATIONS |
| 3-layer caching | `be/agent_tools.py` - ToolExecutor._check_frontend_context() |
| LLM client (function calling) | `be/llm_client.py` - AgentLLMClient |
| Frontend SSE parser | `fe/app.js` - parseSSEBuffer(), sendChatMessage() |
| Frontend caching strategy | `fe/app.js` - stockCache object |
| Chart implementation | `fe/app.js` - chartState, drawChart functions |
| RAG pipeline / FAISS | `be/rag_pipeline.py` |
| Sentiment bias corrections | `be/sentiment_service.py` - aggregate calculation |
| Social scrapers | `be/social_scrapers.py` |
| Design system | `.design-engineer/system.md` |

## Development Rules

1. Keep frontend vanilla JS - no frameworks
2. Respect caching TTLs (see stockCache in app.js)
3. FAISS namespaces: `news:` for articles, `sentiment:` for posts
4. FinBERT has bullish bias - see sentiment_service.py for corrections
5. Tool responses use `role="tool"` in Gemini API (not "user")
6. Embeddings use `result.embeddings[0].values` with new google-genai SDK
7. `chat_service.py` is legacy — all new chat work goes through `agent_service.py`
