from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from polygon_api import PolygonAPI
from chat_routes import register_chat_routes
import os
import atexit

app = Flask(__name__, static_folder='../fe', static_url_path='')
CORS(app)

polygon = PolygonAPI()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/company_tickers.json')
def get_tickers():
    return send_from_directory('../', 'company_tickers.json')

@app.route('/api/ticker/<ticker>/details', methods=['GET'])
def get_ticker_details(ticker):
    try:
        data = polygon.get_ticker_details(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/previous-close', methods=['GET'])
def get_previous_close(ticker):
    try:
        data = polygon.get_previous_close(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/aggregates', methods=['GET'])
def get_aggregates(ticker):
    try:
        from_date = request.args.get('from')
        to_date = request.args.get('to')
        timespan = request.args.get('timespan', 'day')

        data = polygon.get_aggregates(ticker.upper(), timespan, from_date, to_date)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/news', methods=['GET'])
def get_news(ticker):
    try:
        limit = request.args.get('limit', 10, type=int)
        data = polygon.get_ticker_news(ticker.upper(), limit)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/financials', methods=['GET'])
def get_financials(ticker):
    try:
        data = polygon.get_financials(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/snapshot', methods=['GET'])
def get_snapshot(ticker):
    try:
        data = polygon.get_snapshot(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/dividends', methods=['GET'])
def get_dividends(ticker):
    try:
        limit = request.args.get('limit', 10, type=int)
        data = polygon.get_dividends(ticker.upper(), limit)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/splits', methods=['GET'])
def get_splits(ticker):
    try:
        limit = request.args.get('limit', 10, type=int)
        data = polygon.get_splits(ticker.upper(), limit)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ticker/<ticker>/related', methods=['GET'])
def get_related(ticker):
    try:
        data = polygon.get_related_companies(ticker.upper())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/market-status', methods=['GET'])
def get_market_status():
    try:
        data = polygon.get_market_status()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Register chat routes
register_chat_routes(app)

# Graceful shutdown handler for FAISS
from chat_routes import chat_service

def shutdown_handler():
    """Save FAISS index on graceful shutdown"""
    print("Shutting down gracefully, saving FAISS index...")
    try:
        chat_service.vector_store.save()
        print("FAISS index saved successfully")
    except Exception as e:
        print(f"Error saving FAISS index on shutdown: {e}")

atexit.register(shutdown_handler)

if __name__ == '__main__':
    from config import PORT
    app.run(debug=True, port=PORT)
