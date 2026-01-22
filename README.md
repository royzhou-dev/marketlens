# MarketLens - Stock Trading Assistant

A personal stock trading assistant that provides real-time market information and AI-powered insights. Users can explore stock data, news, earnings, and performance metrics powered by the Polygon API.

## Project Structure

```
MarketLens/
├── be/                          # Python backend
│   ├── app.py                   # Flask application
│   ├── config.py                # Configuration management
│   ├── polygon_api.py           # Polygon API client
│   └── requirements.txt         # Python dependencies
├── fe/                          # JavaScript frontend
│   ├── index.html              # Main HTML page
│   ├── app.js                  # Frontend logic
│   └── styles.css              # Styling
├── company_tickers.json        # List of stock tickers
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Features

- **Stock Selection**: Search and select from thousands of stock tickers
- **Real-time Data**: View current market data including price, volume, and market cap
- **Price Charts**: Interactive charts with multiple timeframes (1M, 3M, 6M, 1Y, 5Y)
- **Financial Data**: Access quarterly and annual financial statements
- **News Feed**: Latest news articles related to selected stocks
- **Clean UI**: Simple, responsive interface built with vanilla JavaScript

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Polygon API key (get one at [polygon.io](https://polygon.io/))

### 1. Install Python Dependencies

Navigate to the project directory and install the required Python packages:

```bash
cd "MarketLens"
pip install -r be/requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root directory:

```bash
cp .env.example .env
```

Edit the `.env` file and add your Polygon API key:

```
POLYGON_API_KEY=your_actual_api_key_here
PORT=5000
```

**Important**: Get your free Polygon API key from [https://polygon.io/](https://polygon.io/)

### 3. Run the Application

Start the Flask backend server:

```bash
cd be
python app.py
```

The application will start on `http://localhost:5000`

### 4. Access the Application

Open your web browser and navigate to:

```
http://localhost:5000
```

## Usage

1. **Search for a Stock**: Use the search box to filter stocks by ticker symbol or company name
2. **Select a Stock**: Click on a stock from the dropdown list
3. **Explore Data**: Navigate through different tabs:
   - **Overview**: Key metrics and company description
   - **Chart**: Price performance over various timeframes
   - **Financials**: Quarterly and annual financial statements
   - **News**: Latest news articles about the company

## API Endpoints

The backend provides the following REST API endpoints:

- `GET /api/ticker/<ticker>/details` - Get detailed ticker information
- `GET /api/ticker/<ticker>/previous-close` - Get previous day's close data
- `GET /api/ticker/<ticker>/aggregates` - Get historical price data
- `GET /api/ticker/<ticker>/news` - Get news articles
- `GET /api/ticker/<ticker>/financials` - Get financial statements
- `GET /api/ticker/<ticker>/snapshot` - Get current market snapshot

## Technology Stack

### Backend
- **Flask**: Lightweight Python web framework
- **Requests**: HTTP library for API calls
- **python-dotenv**: Environment variable management
- **Flask-CORS**: Cross-origin resource sharing

### Frontend
- **Vanilla JavaScript**: No frameworks, pure JavaScript
- **HTML5 Canvas**: For rendering price charts
- **CSS3**: Modern styling with flexbox and grid

## Polygon API

This application uses the Polygon API to fetch:
- Real-time and historical stock prices
- Company information and details
- Financial statements
- News articles
- Market snapshots

API Documentation: [https://polygon.io/docs](https://polygon.io/docs)

## Development

### Running in Development Mode

The Flask server runs in debug mode by default, which provides:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

### Environment Variables

- `POLYGON_API_KEY`: Your Polygon API key (required)
- `PORT`: Server port (default: 5000)

## Troubleshooting

### API Key Issues
- Ensure your `.env` file is in the project root directory
- Verify your API key is valid at [polygon.io](https://polygon.io/)
- Check that the API key is properly set in the `.env` file

### CORS Errors
- Make sure Flask-CORS is installed: `pip install Flask-CORS`
- The backend should be running on `http://localhost:5000`

### No Data Displayed
- Check browser console for error messages
- Verify the backend server is running
- Ensure your Polygon API key has sufficient permissions

## License

This is a personal project for educational and personal use.

## Future Enhancements

- AI-powered stock analysis and recommendations
- Portfolio tracking
- Real-time price updates with WebSockets
- Advanced charting with technical indicators
- User authentication and saved preferences
