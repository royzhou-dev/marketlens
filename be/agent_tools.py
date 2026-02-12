"""
Tool definitions and executor for the ReAct agent.
Maps Gemini function declarations to existing backend services.
Includes hybrid caching: frontend context -> server cache -> API call.
"""

import time
import logging
from datetime import datetime, timedelta

from polygon_api import PolygonAPI
from rag_pipeline import ContextRetriever
from sentiment_service import get_sentiment_service
from forecast_service import get_forecast_service

logger = logging.getLogger(__name__)


# -- Tool Schemas (Gemini function declarations) --

TOOL_DECLARATIONS = [
    {
        "name": "get_stock_quote",
        "description": "Get the most recent closing price, open, high, low, and volume for a stock ticker. Use this when the user asks about current price, today's price, or recent trading data.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_company_info",
        "description": "Get detailed company information including name, description, market cap, sector, industry, and exchange. Use this when the user asks about what a company does, its sector, or general company details.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_financials",
        "description": "Get recent financial statements including revenue, net income, gross profit, total assets, and liabilities. Returns the last 4 filing periods. Use this for questions about earnings, revenue, profitability, or balance sheet.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_news",
        "description": "Get recent news articles about a stock. Returns headlines, sources, dates, and descriptions. Use this when the user asks about recent news, headlines, or events.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of articles to return (default 10, max 20)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "search_knowledge_base",
        "description": "Semantic search over previously indexed news articles and research. Use this when the user asks about a specific topic and you need in-depth article content beyond headlines.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query"
                },
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol to filter results"
                }
            },
            "required": ["query", "ticker"]
        }
    },
    {
        "name": "analyze_sentiment",
        "description": "Analyze social media sentiment for a stock by scraping StockTwits, Reddit, and Twitter posts and running FinBERT analysis. This operation takes 10-30 seconds. Use when the user asks about sentiment, social media buzz, or what people think about a stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_price_forecast",
        "description": "Get an LSTM neural network price forecast for the next 30 trading days. May take 30-60 seconds if the model needs training. Use when the user asks about price predictions or forecasts.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_dividends",
        "description": "Get dividend payment history including ex-dividend dates, pay dates, and amounts. Use when the user asks about dividends, yield, or dividend history.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of dividend records to return (default 10)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_stock_splits",
        "description": "Get stock split history including execution dates and split ratios. Use when the user asks about stock splits.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "get_price_history",
        "description": "Get historical OHLCV price data for a date range. Use when the user asks about price trends, historical performance, or needs to compare prices between dates.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. AAPL"
                },
                "from_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "to_date": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format"
                },
                "timespan": {
                    "type": "string",
                    "description": "Time interval for each bar: day, week, or month (default: day)"
                }
            },
            "required": ["ticker", "from_date", "to_date"]
        }
    }
]


class ToolCache:
    """Server-side TTL cache for tool results (Layer 2)."""

    def __init__(self, ttl_seconds=300):
        self._cache = {}
        self._ttl = ttl_seconds

    def get(self, key):
        entry = self._cache.get(key)
        if entry and time.time() - entry["ts"] < self._ttl:
            return entry["data"]
        return None

    def set(self, key, data):
        self._cache[key] = {"data": data, "ts": time.time()}


class ToolExecutor:
    """
    Executes tool calls using a 3-layer cache strategy:
    Layer 1: Frontend context (passed per-request via set_context)
    Layer 2: Server-side TTL cache (ToolCache, 5-min TTL)
    Layer 3: Live API call (last resort)
    """

    def __init__(self, polygon_api, context_retriever, vector_store):
        self.polygon = polygon_api
        self.context_retriever = context_retriever
        self.vector_store = vector_store
        self.sentiment_service = get_sentiment_service(vector_store)
        self.forecast_service = get_forecast_service()
        self.server_cache = ToolCache(ttl_seconds=300)

        # Frontend context for the current request (Layer 1)
        self._frontend_context = {}
        self._context_ticker = None

        self._handlers = {
            "get_stock_quote": self._get_stock_quote,
            "get_company_info": self._get_company_info,
            "get_financials": self._get_financials,
            "get_news": self._get_news,
            "search_knowledge_base": self._search_knowledge_base,
            "analyze_sentiment": self._analyze_sentiment,
            "get_price_forecast": self._get_price_forecast,
            "get_dividends": self._get_dividends,
            "get_stock_splits": self._get_stock_splits,
            "get_price_history": self._get_price_history,
        }

    def set_context(self, frontend_context, ticker):
        """Prime Layer 1 cache with frontend-provided data for this request."""
        self._frontend_context = frontend_context or {}
        self._context_ticker = ticker.upper() if ticker else None

    def execute(self, tool_name, args):
        handler = self._handlers.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**args)
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"error": f"Tool execution failed: {str(e)}"}

    def _check_frontend_context(self, tool_name, ticker):
        """Layer 1: Check if frontend already sent this data."""
        if ticker.upper() != self._context_ticker:
            return None

        mapping = {
            "get_stock_quote": "previousClose",
            "get_company_info": "details",
            "get_financials": "financials",
            "get_news": "news",
            "get_dividends": "dividends",
            "get_stock_splits": "splits",
            "analyze_sentiment": "sentiment",
        }

        context_key = mapping.get(tool_name)
        if not context_key:
            return None

        overview = self._frontend_context.get("overview", {})

        # Some keys are nested under overview
        if context_key in ("previousClose", "details"):
            data = overview.get(context_key)
        else:
            data = self._frontend_context.get(context_key)

        return data if data else None

    def _check_server_cache(self, tool_name, ticker):
        """Layer 2: Check server-side TTL cache."""
        return self.server_cache.get(f"{tool_name}:{ticker}")

    def _cache_result(self, tool_name, ticker, result):
        """Store result in server-side cache."""
        self.server_cache.set(f"{tool_name}:{ticker}", result)

    # -- Tool Handlers --

    def _get_stock_quote(self, ticker):
        ticker = ticker.upper()

        # Layer 1: Frontend context
        fe_data = self._check_frontend_context("get_stock_quote", ticker)
        if fe_data:
            results = fe_data.get("results", [])
            if results:
                r = results[0]
                return {
                    "ticker": ticker,
                    "close": r.get("c"),
                    "open": r.get("o"),
                    "high": r.get("h"),
                    "low": r.get("l"),
                    "volume": r.get("v"),
                    "vwap": r.get("vw"),
                    "source": "cached"
                }

        # Layer 2: Server cache
        cached = self._check_server_cache("get_stock_quote", ticker)
        if cached:
            return cached

        # Layer 3: API call
        data = self.polygon.get_previous_close(ticker)
        results = data.get("results", [])
        if not results:
            return {"error": "No quote data available"}

        r = results[0]
        result = {
            "ticker": ticker,
            "close": r.get("c"),
            "open": r.get("o"),
            "high": r.get("h"),
            "low": r.get("l"),
            "volume": r.get("v"),
            "vwap": r.get("vw"),
        }
        self._cache_result("get_stock_quote", ticker, result)
        return result

    def _get_company_info(self, ticker):
        ticker = ticker.upper()

        fe_data = self._check_frontend_context("get_company_info", ticker)
        if fe_data:
            r = fe_data.get("results", fe_data)
            return {
                "ticker": ticker,
                "name": r.get("name"),
                "description": r.get("description", "")[:500],
                "market_cap": r.get("market_cap"),
                "sector": r.get("sic_description"),
                "homepage_url": r.get("homepage_url"),
                "total_employees": r.get("total_employees"),
                "source": "cached"
            }

        cached = self._check_server_cache("get_company_info", ticker)
        if cached:
            return cached

        data = self.polygon.get_ticker_details(ticker)
        r = data.get("results", {})
        if not r:
            return {"error": "No company data available"}

        result = {
            "ticker": ticker,
            "name": r.get("name"),
            "description": r.get("description", "")[:500],
            "market_cap": r.get("market_cap"),
            "sector": r.get("sic_description"),
            "homepage_url": r.get("homepage_url"),
            "total_employees": r.get("total_employees"),
        }
        self._cache_result("get_company_info", ticker, result)
        return result

    def _get_financials(self, ticker):
        ticker = ticker.upper()

        fe_data = self._check_frontend_context("get_financials", ticker)
        if fe_data:
            return self._format_financials(ticker, fe_data)

        cached = self._check_server_cache("get_financials", ticker)
        if cached:
            return cached

        data = self.polygon.get_financials(ticker)
        result = self._format_financials(ticker, data)
        self._cache_result("get_financials", ticker, result)
        return result

    def _format_financials(self, ticker, data):
        results = data.get("results", [])
        if not results:
            return {"error": "No financial data available"}

        periods = []
        for r in results[:4]:
            financials = r.get("financials", {})
            income = financials.get("income_statement", {})
            balance = financials.get("balance_sheet", {})

            periods.append({
                "period": f"{r.get('fiscal_period', '')} {r.get('fiscal_year', '')}",
                "revenue": income.get("revenues", {}).get("value"),
                "net_income": income.get("net_income_loss", {}).get("value"),
                "gross_profit": income.get("gross_profit", {}).get("value"),
                "total_assets": balance.get("assets", {}).get("value"),
                "total_liabilities": balance.get("liabilities", {}).get("value"),
            })

        return {"ticker": ticker, "periods": periods}

    def _get_news(self, ticker, limit=10):
        ticker = ticker.upper()
        limit = min(limit or 10, 20)

        fe_data = self._check_frontend_context("get_news", ticker)
        if fe_data:
            return self._format_news(ticker, fe_data)

        cached = self._check_server_cache("get_news", ticker)
        if cached:
            return cached

        data = self.polygon.get_ticker_news(ticker, limit=limit)
        result = self._format_news(ticker, data)
        self._cache_result("get_news", ticker, result)
        return result

    def _format_news(self, ticker, data):
        articles = data.get("results", data if isinstance(data, list) else [])
        if not articles:
            return {"error": "No news articles available"}

        formatted = []
        for a in articles[:10]:
            formatted.append({
                "title": a.get("title", ""),
                "source": a.get("publisher", {}).get("name", "Unknown") if isinstance(a.get("publisher"), dict) else a.get("publisher", "Unknown"),
                "published": a.get("published_utc", "")[:10],
                "description": a.get("description", "")[:200],
                "url": a.get("article_url", ""),
            })

        return {"ticker": ticker, "articles": formatted}

    def _search_knowledge_base(self, query, ticker):
        """Search FAISS vector store — no caching, always live search."""
        ticker = ticker.upper()
        contexts = self.context_retriever.retrieve_context(query, ticker)

        if not contexts:
            return {"message": "No relevant articles found in knowledge base. Try using get_news for recent headlines."}

        results = []
        for ctx in contexts[:5]:
            meta = ctx["metadata"]
            results.append({
                "title": meta.get("title", "Untitled"),
                "source": meta.get("source", "Unknown"),
                "date": meta.get("published_date", "")[:10],
                "content": meta.get("full_content", meta.get("content_preview", ""))[:500],
                "relevance_score": round(ctx["score"], 3),
            })

        return {"ticker": ticker, "results": results}

    def _analyze_sentiment(self, ticker):
        ticker = ticker.upper()

        # Layer 1: Frontend context (already-analyzed sentiment)
        fe_data = self._check_frontend_context("analyze_sentiment", ticker)
        if fe_data:
            aggregate = fe_data.get("aggregate", fe_data)
            posts = fe_data.get("posts", [])
            return {
                "ticker": ticker,
                "overall_sentiment": aggregate.get("label"),
                "score": aggregate.get("score"),
                "confidence": aggregate.get("confidence"),
                "post_count": aggregate.get("post_count"),
                "sources": aggregate.get("sources", {}),
                "top_posts": [
                    {
                        "platform": p.get("platform"),
                        "content": p.get("content", "")[:200],
                        "sentiment": p.get("sentiment", {}).get("label", p.get("sentiment_label", "")),
                    }
                    for p in posts[:5]
                ],
                "source": "cached"
            }

        cached = self._check_server_cache("analyze_sentiment", ticker)
        if cached:
            return cached

        # Layer 3: Live scrape + analysis (slow, 10-30s)
        data = self.sentiment_service.analyze_ticker(ticker)
        aggregate = data.get("aggregate", {})
        posts = data.get("posts", [])

        result = {
            "ticker": ticker,
            "overall_sentiment": aggregate.get("label"),
            "score": aggregate.get("score"),
            "confidence": aggregate.get("confidence"),
            "post_count": aggregate.get("post_count"),
            "sources": aggregate.get("sources", {}),
            "top_posts": [
                {
                    "platform": p.get("platform"),
                    "content": p.get("content", "")[:200],
                    "sentiment": p.get("sentiment", {}).get("label", ""),
                }
                for p in posts[:5]
            ],
        }
        self._cache_result("analyze_sentiment", ticker, result)
        return result

    def _get_price_forecast(self, ticker):
        ticker = ticker.upper()

        cached = self._check_server_cache("get_price_forecast", ticker)
        if cached:
            return cached

        data = self.forecast_service.get_forecast(ticker)
        if "error" in data:
            return {"error": data["error"]}

        forecast = data.get("forecast", [])
        result = {
            "ticker": ticker,
            "predictions": [
                {
                    "date": f.get("date"),
                    "predicted_close": f.get("predicted_close"),
                    "upper_bound": f.get("upper_bound"),
                    "lower_bound": f.get("lower_bound"),
                }
                for f in forecast[:10]  # First 10 days to keep context manageable
            ],
            "model_info": data.get("model_info", {}),
        }
        self._cache_result("get_price_forecast", ticker, result)
        return result

    def _get_dividends(self, ticker, limit=10):
        ticker = ticker.upper()

        fe_data = self._check_frontend_context("get_dividends", ticker)
        if fe_data:
            return self._format_dividends(ticker, fe_data)

        cached = self._check_server_cache("get_dividends", ticker)
        if cached:
            return cached

        data = self.polygon.get_dividends(ticker, limit=limit or 10)
        result = self._format_dividends(ticker, data)
        self._cache_result("get_dividends", ticker, result)
        return result

    def _format_dividends(self, ticker, data):
        results = data.get("results", data if isinstance(data, list) else [])
        if not results:
            return {"message": "No dividend data available for this ticker."}

        formatted = []
        for d in results[:10]:
            formatted.append({
                "ex_date": d.get("ex_dividend_date", ""),
                "pay_date": d.get("pay_date", ""),
                "amount": d.get("cash_amount"),
                "frequency": d.get("frequency"),
            })

        return {"ticker": ticker, "dividends": formatted}

    def _get_stock_splits(self, ticker):
        ticker = ticker.upper()

        fe_data = self._check_frontend_context("get_stock_splits", ticker)
        if fe_data:
            return self._format_splits(ticker, fe_data)

        cached = self._check_server_cache("get_stock_splits", ticker)
        if cached:
            return cached

        data = self.polygon.get_splits(ticker)
        result = self._format_splits(ticker, data)
        self._cache_result("get_stock_splits", ticker, result)
        return result

    def _format_splits(self, ticker, data):
        results = data.get("results", data if isinstance(data, list) else [])
        if not results:
            return {"message": "No stock split history found for this ticker."}

        formatted = []
        for s in results[:10]:
            formatted.append({
                "execution_date": s.get("execution_date", ""),
                "split_from": s.get("split_from"),
                "split_to": s.get("split_to"),
                "ratio": f"{s.get('split_to', 1)}-for-{s.get('split_from', 1)}",
            })

        return {"ticker": ticker, "splits": formatted}

    def _get_price_history(self, ticker, from_date, to_date, timespan=None):
        ticker = ticker.upper()
        timespan = timespan or "day"

        cache_key = f"get_price_history:{ticker}:{from_date}:{to_date}:{timespan}"
        cached = self.server_cache.get(cache_key)
        if cached:
            return cached

        data = self.polygon.get_aggregates(ticker, timespan=timespan, from_date=from_date, to_date=to_date)
        results = data.get("results", [])
        if not results:
            return {"error": "No price history available for the given date range"}

        formatted = []
        for bar in results:
            formatted.append({
                "date": datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d"),
                "open": bar.get("o"),
                "high": bar.get("h"),
                "low": bar.get("l"),
                "close": bar.get("c"),
                "volume": bar.get("v"),
            })

        result = {
            "ticker": ticker,
            "timespan": timespan,
            "from": from_date,
            "to": to_date,
            "bars": formatted,
            "count": len(formatted),
        }
        self.server_cache.set(cache_key, result)
        return result
