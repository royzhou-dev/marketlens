from flask import request, jsonify, Response, stream_with_context
from agent_service import AgentService
import json

# Initialize agent service
agent_service = AgentService()


def format_sse(event_type, data):
    """Format a Server-Sent Event."""
    if isinstance(data, dict):
        data_str = json.dumps(data)
    else:
        data_str = str(data)
    return f"event: {event_type}\ndata: {data_str}\n\n"


def register_chat_routes(app):
    """Register chat-related routes with Flask app"""

    @app.route('/api/chat/message', methods=['POST'])
    def chat_message():
        """
        Main chat endpoint - processes user messages via agent and streams responses

        Request body:
        {
            "ticker": "AAPL",
            "message": "What's the latest news?",
            "context": {
                "overview": {...},
                "financials": {...},
                "news": [...],
                ...
            },
            "conversation_id": "uuid"
        }

        Response: Server-Sent Events stream with typed events:
          event: tool_call  - Agent is calling a tool
          event: text       - Response text chunk
          event: done       - Stream complete
          event: error      - Error occurred
        """
        try:
            data = request.get_json()

            ticker = data.get('ticker')
            message = data.get('message')
            context = data.get('context', {})
            conversation_id = data.get('conversation_id', 'default')

            if not ticker or not message:
                return jsonify({"error": "Missing required fields"}), 400

            def generate():
                """Generator for structured SSE streaming."""
                try:
                    for event_type, event_data in agent_service.process_message(
                        ticker=ticker,
                        message=message,
                        frontend_context=context,
                        conversation_id=conversation_id
                    ):
                        yield format_sse(event_type, event_data)
                except Exception as e:
                    print(f"Error in stream: {e}")
                    yield format_sse("error", {"message": str(e)})

            return Response(
                stream_with_context(generate()),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )

        except Exception as e:
            print(f"Error in chat_message: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chat/scrape-articles', methods=['POST'])
    def scrape_articles():
        """
        Background job to scrape and embed articles

        Request body:
        {
            "ticker": "AAPL",
            "articles": [...]
        }

        Response:
        {
            "scraped": 8,
            "embedded": 8,
            "failed": 2,
            "skipped": 5
        }
        """
        try:
            data = request.get_json()

            ticker = data.get('ticker')
            articles = data.get('articles', [])

            if not ticker:
                return jsonify({"error": "Missing ticker"}), 400

            if not articles:
                return jsonify({
                    "scraped": 0,
                    "embedded": 0,
                    "failed": 0,
                    "skipped": 0,
                    "message": "No articles provided"
                }), 200

            results = agent_service.scrape_and_embed_articles(ticker, articles)

            return jsonify(results), 200

        except Exception as e:
            print(f"Error in scrape_articles: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chat/conversations/<conversation_id>', methods=['GET'])
    def get_conversation(conversation_id):
        """Get conversation history"""
        try:
            history = agent_service.conversation_manager.get_history(conversation_id)
            return jsonify({"messages": history}), 200
        except Exception as e:
            print(f"Error in get_conversation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chat/clear/<conversation_id>', methods=['DELETE'])
    def clear_conversation(conversation_id):
        """Clear conversation history"""
        try:
            agent_service.conversation_manager.clear_conversation(conversation_id)
            return jsonify({"success": True}), 200
        except Exception as e:
            print(f"Error in clear_conversation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/api/chat/health', methods=['GET'])
    def chat_health():
        """Health check endpoint for chat service"""
        try:
            return jsonify({
                "status": "healthy",
                "components": {
                    "agent": "ok",
                    "embeddings": "ok",
                    "vector_store": "ok",
                    "llm": "ok"
                }
            }), 200
        except Exception as e:
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    @app.route('/api/chat/debug/chunks', methods=['GET'])
    def debug_chunks():
        """Debug endpoint to view all stored RAG chunks"""
        try:
            ticker_filter = request.args.get('ticker')
            limit = int(request.args.get('limit', 50))

            vector_store = agent_service.vector_store
            all_metadata = vector_store.metadata

            chunks = []
            for internal_id, meta in all_metadata.items():
                if ticker_filter and meta.get('ticker') != ticker_filter:
                    continue

                chunks.append({
                    "id": internal_id,
                    "doc_id": meta.get('doc_id', ''),
                    "ticker": meta.get('ticker', ''),
                    "type": meta.get('type', ''),
                    "title": meta.get('title', ''),
                    "url": meta.get('url', ''),
                    "source": meta.get('source', ''),
                    "published_date": meta.get('published_date', ''),
                    "content_preview": meta.get('content_preview', ''),
                    "full_content": meta.get('full_content', '')
                })

                if len(chunks) >= limit:
                    break

            return jsonify({
                "total": len(all_metadata),
                "returned": len(chunks),
                "index_stats": vector_store.get_stats(),
                "chunks": chunks
            }), 200

        except Exception as e:
            print(f"Error in debug_chunks: {e}")
            return jsonify({"error": str(e)}), 500
