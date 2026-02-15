const API_BASE = 'http://localhost:5000/api';
let allTickers = [];
let currentTicker = '';
let chartInstance = null;
let highlightedIndex = -1;
let dropdownItems = [];

// Cache TTL configuration (in minutes)
const CACHE_TTL = {
    STATIC: null,           // Cache until page refresh
    DAILY: 1440,            // 24 hours (for EOD data)
    MODERATE: 30,           // 30 minutes
    SHORT: 15               // 15 minutes
};

// Cache manager
const cache = {
    data: {},

    set(key, value, ttlMinutes = null) {
        this.data[key] = {
            value: value,
            timestamp: Date.now(),
            ttl: ttlMinutes ? ttlMinutes * 60 * 1000 : null
        };
    },

    get(key) {
        const item = this.data[key];
        if (!item) return null;

        // Check if expired
        if (item.ttl && (Date.now() - item.timestamp > item.ttl)) {
            delete this.data[key];
            return null;
        }

        return item.value;
    },

    has(key) {
        return this.get(key) !== null;
    },

    clear() {
        this.data = {};
    },

    getStats() {
        return {
            entries: Object.keys(this.data).length,
            keys: Object.keys(this.data)
        };
    }
};

// Expose cache to global scope for debugging
window.stockCache = cache;

// Recent/Popular Ticker Functions
function getRecentTickers() {
    try {
        const recent = localStorage.getItem('recentTickers');
        if (recent) {
            return JSON.parse(recent);
        }
    } catch (error) {
        console.error('Error reading recent tickers:', error);
    }
    return [];
}

function saveRecentTicker(ticker, title) {
    try {
        let recent = getRecentTickers();

        // Remove if already exists
        recent = recent.filter(item => item.ticker !== ticker);

        // Add to front
        recent.unshift({ ticker, title });

        // Keep only 5 most recent
        recent = recent.slice(0, 5);

        localStorage.setItem('recentTickers', JSON.stringify(recent));
    } catch (error) {
        console.error('Error saving recent ticker:', error);
    }
}

function getPopularTickers() {
    const popularSymbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD'];
    return allTickers.filter(ticker => popularSymbols.includes(ticker.ticker));
}

// Dropdown control functions
function showDropdown() {
    document.getElementById('dropdownList').classList.remove('hidden');
}

function hideDropdown() {
    document.getElementById('dropdownList').classList.add('hidden');
    highlightedIndex = -1;
}

function populateDropdown(tickers, searchTerm = '') {
    const dropdown = document.getElementById('dropdownList');
    dropdown.innerHTML = '';
    dropdownItems = [];

    if (searchTerm === '') {
        // Show recent and popular tickers
        const recent = getRecentTickers();
        const popular = getPopularTickers();

        if (recent.length > 0) {
            const recentHeader = document.createElement('div');
            recentHeader.className = 'dropdown-section-header';
            recentHeader.textContent = 'Recent';
            dropdown.appendChild(recentHeader);

            recent.forEach(item => {
                const div = createDropdownItem(item.ticker, item.title);
                dropdown.appendChild(div);
                dropdownItems.push({ element: div, ticker: item.ticker, title: item.title });
            });
        }

        if (popular.length > 0) {
            const popularHeader = document.createElement('div');
            popularHeader.className = 'dropdown-section-header';
            popularHeader.textContent = 'Popular';
            dropdown.appendChild(popularHeader);

            popular.forEach(item => {
                const div = createDropdownItem(item.ticker, item.title);
                dropdown.appendChild(div);
                dropdownItems.push({ element: div, ticker: item.ticker, title: item.title });
            });
        }
    } else {
        // Show filtered results
        const limited = tickers.slice(0, 50);

        if (limited.length === 0) {
            const noResults = document.createElement('div');
            noResults.className = 'dropdown-no-results';
            noResults.textContent = 'No results found';
            dropdown.appendChild(noResults);
        } else {
            limited.forEach(item => {
                const div = createDropdownItem(item.ticker, item.title);
                dropdown.appendChild(div);
                dropdownItems.push({ element: div, ticker: item.ticker, title: item.title });
            });
        }
    }
}

function createDropdownItem(ticker, title) {
    const div = document.createElement('div');
    div.className = 'dropdown-item';
    div.textContent = `${ticker} - ${title}`;
    div.dataset.ticker = ticker;
    div.dataset.title = title;

    // Use mousedown instead of click to fire before blur
    div.addEventListener('mousedown', (e) => {
        e.preventDefault();
        selectTicker(ticker, title);
    });

    return div;
}

function selectTicker(ticker, title) {
    const input = document.getElementById('tickerSearch');
    input.value = `${ticker} - ${title}`;
    currentTicker = ticker;
    hideDropdown();
    saveRecentTicker(ticker, title);
    loadStockData(ticker);
    updateChatContext();
}

function highlightItem(index) {
    // Remove all highlights
    dropdownItems.forEach(item => item.element.classList.remove('highlighted'));

    if (index >= 0 && index < dropdownItems.length) {
        dropdownItems[index].element.classList.add('highlighted');
        dropdownItems[index].element.scrollIntoView({ block: 'nearest' });
    }
}

// Load tickers on page load
document.addEventListener('DOMContentLoaded', async () => {
    initTheme();
    await loadTickers();
    await loadMarketStatus();
    setupEventListeners();
});

// Theme toggle functionality
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

    if (savedTheme) {
        document.documentElement.setAttribute('data-theme', savedTheme);
    } else if (prefersDark) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }

    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', toggleTheme);
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Redraw chart with new theme colors
    if (chartState.data) {
        drawChart(1);
    }
}

async function loadTickers() {
    try {
        const response = await fetch('../company_tickers.json');
        const data = await response.json();

        allTickers = Object.values(data).map(item => ({
            ticker: item.ticker,
            title: item.title,
            cik: item.cik_str
        }));
    } catch (error) {
        console.error('Error loading tickers:', error);
    }
}

function setupEventListeners() {
    const tickerSearch = document.getElementById('tickerSearch');
    const dropdownContainer = document.querySelector('.dropdown-container');

    // Input focus - show dropdown with recent/popular or current results
    tickerSearch.addEventListener('focus', (e) => {
        const searchTerm = e.target.value.trim();
        if (searchTerm === '') {
            populateDropdown([], '');
        } else {
            // Select all text for easy replacement
            e.target.select();
            const filtered = allTickers.filter(item =>
                item.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
                item.title.toLowerCase().includes(searchTerm.toLowerCase())
            );
            populateDropdown(filtered, searchTerm);
        }
        showDropdown();
    });

    // Input blur - hide dropdown with delay
    tickerSearch.addEventListener('blur', () => {
        setTimeout(() => {
            hideDropdown();
        }, 200);
    });

    // Input keydown - handle keyboard navigation
    tickerSearch.addEventListener('keydown', (e) => {
        const dropdown = document.getElementById('dropdownList');
        const isOpen = !dropdown.classList.contains('hidden');

        if (!isOpen && e.key !== 'Escape') return;

        switch(e.key) {
            case 'Escape':
                if (e.target.value) {
                    e.target.value = '';
                    populateDropdown([], '');
                    showDropdown();
                } else {
                    hideDropdown();
                }
                break;

            case 'ArrowDown':
                e.preventDefault();
                highlightedIndex++;
                if (highlightedIndex >= dropdownItems.length) {
                    highlightedIndex = 0;
                }
                highlightItem(highlightedIndex);
                break;

            case 'ArrowUp':
                e.preventDefault();
                highlightedIndex--;
                if (highlightedIndex < 0) {
                    highlightedIndex = dropdownItems.length - 1;
                }
                highlightItem(highlightedIndex);
                break;

            case 'Enter':
                e.preventDefault();
                if (highlightedIndex >= 0 && highlightedIndex < dropdownItems.length) {
                    const item = dropdownItems[highlightedIndex];
                    selectTicker(item.ticker, item.title);
                }
                break;
        }
    });

    // Input input - filter and show results
    tickerSearch.addEventListener('input', (e) => {
        const searchTerm = e.target.value.trim();
        highlightedIndex = -1;

        if (searchTerm === '') {
            populateDropdown([], '');
        } else {
            const filtered = allTickers.filter(item =>
                item.ticker.toLowerCase().includes(searchTerm.toLowerCase()) ||
                item.title.toLowerCase().includes(searchTerm.toLowerCase())
            );
            populateDropdown(filtered, searchTerm);
        }
        showDropdown();
    });

    // Click outside to close dropdown
    document.addEventListener('click', (e) => {
        if (!dropdownContainer.contains(e.target)) {
            hideDropdown();
        }
    });

    // Tab switching
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.dataset.tab;
            switchTab(tabName);
        });
    });

    // Chart range buttons
    const rangeButtons = document.querySelectorAll('.chart-range-btn');
    rangeButtons.forEach(button => {
        button.addEventListener('click', () => {
            const range = button.dataset.range;
            loadChartData(currentTicker, range);
            rangeButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });

    // Chart view toggle buttons (line/candle)
    const viewButtons = document.querySelectorAll('.chart-view-btn');
    viewButtons.forEach(button => {
        button.addEventListener('click', () => {
            viewButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            chartState.viewMode = button.dataset.view;
            if (chartState.data) {
                drawChart(1);
            }
        });
    });

    // Setup chat listeners
    setupChatListeners();
}

function switchTab(tabName) {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabPanes.forEach(pane => pane.classList.remove('active'));

    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(tabName).classList.add('active');

    if (tabName === 'overview' && currentTicker) {
        loadChartData(currentTicker, document.querySelector('.chart-range-btn.active').dataset.range || '1M');
    } else if (tabName === 'financials' && currentTicker) {
        loadFinancials(currentTicker);
    } else if (tabName === 'news' && currentTicker) {
        loadNews(currentTicker);
    } else if (tabName === 'dividends' && currentTicker) {
        loadDividends(currentTicker);
    } else if (tabName === 'splits' && currentTicker) {
        loadSplits(currentTicker);
    } else if (tabName === 'sentiment' && currentTicker) {
        loadSentiment(currentTicker);
    } else if (tabName === 'forecast' && currentTicker) {
        loadForecast(currentTicker);
    }
}

function clearStockDisplay() {
    document.getElementById('stockTitle').textContent = '';
    document.getElementById('stockPrice').textContent = '';
    document.getElementById('stockPrice').className = 'stock-price';
    document.getElementById('companyDesc').textContent = '';
    document.getElementById('marketCap').textContent = '--';
    document.getElementById('openPrice').textContent = '--';
    document.getElementById('highPrice').textContent = '--';
    document.getElementById('lowPrice').textContent = '--';
    document.getElementById('volume').textContent = '--';
    document.getElementById('peRatio').textContent = '--';
}

async function loadStockData(ticker) {
    currentTicker = ticker;
    document.getElementById('stockData').classList.remove('hidden');
    clearStockDisplay();

    try {
        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        const loadPromises = [
            loadTickerDetails(ticker),
            loadPreviousClose(ticker)
        ];
        if (activeTab === 'overview') {
            loadPromises.push(loadChartData(ticker, '1M'));
        }
        await Promise.all(loadPromises);

        if (activeTab === 'financials') {
            await loadFinancials(ticker);
        } else if (activeTab === 'news') {
            await loadNews(ticker);
        } else if (activeTab === 'sentiment') {
            loadSentiment(ticker);
        } else if (activeTab === 'forecast') {
            loadForecast(ticker);
        } else if (activeTab === 'dividends') {
            loadDividends(ticker);
        } else if (activeTab === 'splits') {
            loadSplits(ticker);
        }

        // Preload news and trigger article scraping for RAG in background
        preloadNewsForRAG(ticker);
    } catch (error) {
        console.error('Error loading stock data:', error);
        alert('Error loading stock data. Please check your API key and try again.');
    }
}

async function loadTickerDetails(ticker) {
    const cacheKey = `details_${ticker}`;

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderTickerDetails(data);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/details`);
        const data = await response.json();

        if (data.results) {
            cache.set(cacheKey, data, CACHE_TTL.STATIC);
            renderTickerDetails(data);
        }
    } catch (error) {
        console.error('Error loading ticker details:', error);
    }
}

function renderTickerDetails(data) {
    if (data.results) {
        const results = data.results;
        document.getElementById('stockTitle').textContent =
            `${results.ticker} - ${results.name}`;
        document.getElementById('companyDesc').textContent =
            results.description || 'No description available';
        document.getElementById('marketCap').textContent =
            results.market_cap ? formatLargeNumber(results.market_cap) : '--';
    }
}

async function loadSnapshot(ticker) {
    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/snapshot`);
        const data = await response.json();

        if (data.ticker) {
            const ticker_data = data.ticker;
            const day = ticker_data.day || {};

            document.getElementById('openPrice').textContent =
                day.o ? `$${day.o.toFixed(2)}` : '--';
            document.getElementById('highPrice').textContent =
                day.h ? `$${day.h.toFixed(2)}` : '--';
            document.getElementById('lowPrice').textContent =
                day.l ? `$${day.l.toFixed(2)}` : '--';
            document.getElementById('volume').textContent =
                day.v ? formatLargeNumber(day.v) : '--';
        }
    } catch (error) {
        console.error('Error loading snapshot:', error);
    }
}

async function loadPreviousClose(ticker) {
    const cacheKey = `prev_close_${ticker}`;

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderPreviousClose(data);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/previous-close`);
        const data = await response.json();

        if (data.results && data.results.length > 0) {
            cache.set(cacheKey, data, CACHE_TTL.DAILY);
            renderPreviousClose(data);
        }
    } catch (error) {
        console.error('Error loading previous close:', error);
    }
}

function renderPreviousClose(data) {
    if (data.results && data.results.length > 0) {
        const result = data.results[0];
        const priceElement = document.getElementById('stockPrice');
        priceElement.textContent = `$${result.c.toFixed(2)}`;

        const change = result.c - result.o;
        const changePercent = ((change / result.o) * 100).toFixed(2);

        if (change >= 0) {
            priceElement.classList.add('positive');
            priceElement.classList.remove('negative');
            priceElement.innerHTML += ` <span style="font-size: 0.6em;">+${changePercent}%</span>`;
        } else {
            priceElement.classList.add('negative');
            priceElement.classList.remove('positive');
            priceElement.innerHTML += ` <span style="font-size: 0.6em;">${changePercent}%</span>`;
        }

        // Populate Overview metrics from previous close data
        document.getElementById('openPrice').textContent = `$${result.o.toFixed(2)}`;
        document.getElementById('highPrice').textContent = `$${result.h.toFixed(2)}`;
        document.getElementById('lowPrice').textContent = `$${result.l.toFixed(2)}`;
        document.getElementById('volume').textContent = formatLargeNumber(result.v);
    }
}

async function loadChartData(ticker, range) {
    const cacheKey = `chart_${ticker}_${range}`;
    const chartLoading = document.getElementById('chartLoading');

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        if (data.results) {
            renderChart(data.results);
        }
        return;
    }

    chartLoading.classList.remove('hidden');
    const { from, to } = getDateRange(range);

    try {
        const response = await fetch(
            `${API_BASE}/ticker/${ticker}/aggregates?from=${from}&to=${to}&timespan=day`
        );
        const data = await response.json();

        if (data.results) {
            cache.set(cacheKey, data, CACHE_TTL.DAILY);
            renderChart(data.results);
        }
    } catch (error) {
        console.error('Error loading chart data:', error);
    } finally {
        chartLoading.classList.add('hidden');
    }
}

function getDateRange(range) {
    const to = new Date();
    const from = new Date();

    switch(range) {
        case '1M':
            from.setMonth(from.getMonth() - 1);
            break;
        case '3M':
            from.setMonth(from.getMonth() - 3);
            break;
        case '6M':
            from.setMonth(from.getMonth() - 6);
            break;
        case '1Y':
            from.setFullYear(from.getFullYear() - 1);
            break;
        case '5Y':
            from.setFullYear(from.getFullYear() - 5);
            break;
    }

    return {
        from: from.toISOString().split('T')[0],
        to: to.toISOString().split('T')[0]
    };
}

// Chart state for interactivity
let chartState = {
    data: null,
    canvas: null,
    ctx: null,
    padding: { top: 20, right: 20, bottom: 40, left: 65 },
    hoveredIndex: -1,
    animationProgress: 0,
    animationFrame: null,
    viewMode: 'line' // 'line' or 'candle'
};

function getChartColors() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    return {
        line: isDark ? '#818cf8' : '#4f46e5',
        lineLight: isDark ? '#a5b4fc' : '#6366f1',
        gradientTop: isDark ? 'rgba(129, 140, 248, 0.3)' : 'rgba(79, 70, 229, 0.15)',
        gradientBottom: isDark ? 'rgba(129, 140, 248, 0)' : 'rgba(79, 70, 229, 0)',
        grid: isDark ? 'rgba(148, 163, 184, 0.1)' : 'rgba(148, 163, 184, 0.3)',
        text: isDark ? '#94a3b8' : '#64748b',
        textStrong: isDark ? '#cbd5e1' : '#475569',
        crosshair: isDark ? 'rgba(148, 163, 184, 0.5)' : 'rgba(100, 116, 139, 0.4)',
        tooltipBg: isDark ? '#1e293b' : '#ffffff',
        tooltipBorder: isDark ? '#334155' : '#e2e8f0',
        positive: '#10b981',
        negative: '#ef4444'
    };
}

function renderChart(data) {
    const canvas = document.getElementById('priceChart');
    const ctx = canvas.getContext('2d');

    // High DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';

    chartState.data = data;
    chartState.canvas = canvas;
    chartState.ctx = ctx;
    chartState.hoveredIndex = -1;

    // Cancel any existing animation
    if (chartState.animationFrame) {
        cancelAnimationFrame(chartState.animationFrame);
    }

    // Animate the chart drawing
    chartState.animationProgress = 0;
    animateChart();

    // Set up mouse events
    canvas.onmousemove = handleChartMouseMove;
    canvas.onmouseleave = handleChartMouseLeave;
}

function animateChart() {
    chartState.animationProgress += 0.04;
    if (chartState.animationProgress > 1) chartState.animationProgress = 1;

    drawChart(chartState.animationProgress);

    if (chartState.animationProgress < 1) {
        chartState.animationFrame = requestAnimationFrame(animateChart);
    }
}

function drawChart(progress = 1) {
    const { data, canvas, ctx, padding, viewMode } = chartState;
    if (!data || !ctx) return;

    const colors = getChartColors();
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const height = canvas.height / dpr;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // For candlestick, use high/low for price range
    let minPrice, maxPrice;
    if (viewMode === 'candle') {
        minPrice = Math.min(...data.map(d => d.l));
        maxPrice = Math.max(...data.map(d => d.h));
    } else {
        const prices = data.map(d => d.c);
        minPrice = Math.min(...prices);
        maxPrice = Math.max(...prices);
    }
    const pricePadding = (maxPrice - minPrice) * 0.05;
    const adjustedMin = minPrice - pricePadding;
    const adjustedMax = maxPrice + pricePadding;
    const priceRange = adjustedMax - adjustedMin;

    ctx.clearRect(0, 0, width, height);

    // Draw horizontal grid lines and Y-axis labels
    const numGridLines = 5;
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = colors.text;
    ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= numGridLines; i++) {
        const y = padding.top + (i / numGridLines) * chartHeight;
        const price = adjustedMax - (i / numGridLines) * priceRange;

        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillText(`$${price.toFixed(2)}`, padding.left - 8, y);
    }

    // Draw chart based on view mode
    if (viewMode === 'candle') {
        drawCandlesticks(data, adjustedMax, priceRange, chartWidth, chartHeight, height, padding, colors, progress);
    } else {
        drawLineChart(data, adjustedMax, priceRange, chartWidth, chartHeight, height, padding, colors, progress);
    }

    // Draw X-axis date labels
    drawXAxisLabels(data, chartWidth, height, padding, colors);

    // Draw crosshair and tooltip if hovering
    if (chartState.hoveredIndex >= 0 && chartState.hoveredIndex < data.length && progress === 1) {
        drawCrosshair(chartState.hoveredIndex, data, adjustedMin, adjustedMax, priceRange, chartWidth, chartHeight, width, height, padding, colors);
    }
}

function drawLineChart(data, adjustedMax, priceRange, chartWidth, chartHeight, height, padding, colors, progress) {
    const ctx = chartState.ctx;

    // Calculate points for animation
    const pointsToDraw = Math.floor(data.length * progress);
    const points = [];

    for (let i = 0; i < pointsToDraw; i++) {
        const x = padding.left + (i / (data.length - 1)) * chartWidth;
        const y = padding.top + ((adjustedMax - data[i].c) / priceRange) * chartHeight;
        points.push({ x, y, data: data[i] });
    }

    if (points.length < 2) return;

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
    gradient.addColorStop(0, colors.gradientTop);
    gradient.addColorStop(1, colors.gradientBottom);

    ctx.beginPath();
    ctx.moveTo(points[0].x, height - padding.bottom);
    points.forEach(p => ctx.lineTo(p.x, p.y));
    ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw the line
    ctx.beginPath();
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    points.forEach((p, i) => {
        if (i === 0) {
            ctx.moveTo(p.x, p.y);
        } else {
            ctx.lineTo(p.x, p.y);
        }
    });
    ctx.stroke();
}

function drawCandlesticks(data, adjustedMax, priceRange, chartWidth, chartHeight, height, padding, colors, progress) {
    const ctx = chartState.ctx;
    const candleCount = data.length;
    const totalCandleSpace = chartWidth / candleCount;
    const candleWidth = Math.max(1, totalCandleSpace * 0.7);
    const candlesToDraw = Math.floor(candleCount * progress);

    for (let i = 0; i < candlesToDraw; i++) {
        const point = data[i];
        const x = padding.left + (i + 0.5) * totalCandleSpace;

        const openY = padding.top + ((adjustedMax - point.o) / priceRange) * chartHeight;
        const closeY = padding.top + ((adjustedMax - point.c) / priceRange) * chartHeight;
        const highY = padding.top + ((adjustedMax - point.h) / priceRange) * chartHeight;
        const lowY = padding.top + ((adjustedMax - point.l) / priceRange) * chartHeight;

        const isUp = point.c >= point.o;
        const candleColor = isUp ? colors.positive : colors.negative;

        // Draw wick (high to low line)
        ctx.beginPath();
        ctx.strokeStyle = candleColor;
        ctx.lineWidth = 1;
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        // Draw body (open to close rectangle)
        const bodyTop = Math.min(openY, closeY);
        const bodyHeight = Math.max(1, Math.abs(closeY - openY));

        ctx.fillStyle = candleColor;
        ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, bodyHeight);
    }
}

function drawXAxisLabels(data, chartWidth, height, padding, colors) {
    const ctx = chartState.ctx;
    const labelCount = Math.min(6, data.length);
    const step = Math.floor(data.length / labelCount);

    ctx.fillStyle = colors.text;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let i = 0; i < data.length; i += step) {
        const x = padding.left + (i / (data.length - 1)) * chartWidth;
        const date = new Date(data[i].t);
        const label = formatDateLabel(date, data.length);
        ctx.fillText(label, x, height - padding.bottom + 8);
    }

    // Always show last date
    const lastX = padding.left + chartWidth;
    const lastDate = new Date(data[data.length - 1].t);
    ctx.fillText(formatDateLabel(lastDate, data.length), lastX, height - padding.bottom + 8);
}

function formatDateLabel(date, dataLength) {
    const month = date.toLocaleDateString('en-US', { month: 'short' });
    const day = date.getDate();
    const year = date.getFullYear().toString().slice(-2);

    if (dataLength > 365) {
        return `${month} '${year}`;
    }
    return `${month} ${day}`;
}

function drawCrosshair(index, data, adjustedMin, adjustedMax, priceRange, chartWidth, chartHeight, width, height, padding, colors) {
    const ctx = chartState.ctx;
    const point = data[index];
    const x = padding.left + (index / (data.length - 1)) * chartWidth;
    const y = padding.top + ((adjustedMax - point.c) / priceRange) * chartHeight;

    // Vertical line
    ctx.strokeStyle = colors.crosshair;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Point dot
    ctx.beginPath();
    ctx.fillStyle = colors.line;
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = colors.tooltipBg;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Tooltip
    drawTooltip(x, y, point, data, index, width, height, padding, colors);
}

function drawTooltip(x, y, point, data, index, width, height, padding, colors) {
    const ctx = chartState.ctx;
    const isCandleMode = chartState.viewMode === 'candle';

    const date = new Date(point.t);
    const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

    const change = index > 0 ? point.c - data[index - 1].c : 0;
    const changePercent = index > 0 ? (change / data[index - 1].c) * 100 : 0;
    const changeColor = change >= 0 ? colors.positive : colors.negative;
    const changeSign = change >= 0 ? '+' : '';

    const tooltipWidth = isCandleMode ? 155 : 140;
    const tooltipHeight = isCandleMode ? 105 : 72;
    let tooltipX = x + 12;
    let tooltipY = y - tooltipHeight / 2;

    // Keep tooltip in bounds
    if (tooltipX + tooltipWidth > width - padding.right) {
        tooltipX = x - tooltipWidth - 12;
    }
    if (tooltipY < padding.top) {
        tooltipY = padding.top;
    }
    if (tooltipY + tooltipHeight > height - padding.bottom) {
        tooltipY = height - padding.bottom - tooltipHeight;
    }

    // Tooltip background
    ctx.fillStyle = colors.tooltipBg;
    ctx.strokeStyle = colors.tooltipBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 6);
    ctx.fill();
    ctx.stroke();

    // Tooltip content
    ctx.fillStyle = colors.text;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(dateStr, tooltipX + 10, tooltipY + 10);

    if (isCandleMode) {
        // OHLC display for candlestick mode
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const ohlcY = tooltipY + 28;
        const lineHeight = 16;

        ctx.fillStyle = colors.text;
        ctx.fillText('O:', tooltipX + 10, ohlcY);
        ctx.fillText('H:', tooltipX + 10, ohlcY + lineHeight);
        ctx.fillText('L:', tooltipX + 10, ohlcY + lineHeight * 2);
        ctx.fillText('C:', tooltipX + 10, ohlcY + lineHeight * 3);

        ctx.fillStyle = colors.textStrong;
        ctx.fillText(`$${point.o.toFixed(2)}`, tooltipX + 28, ohlcY);
        ctx.fillText(`$${point.h.toFixed(2)}`, tooltipX + 28, ohlcY + lineHeight);
        ctx.fillText(`$${point.l.toFixed(2)}`, tooltipX + 28, ohlcY + lineHeight * 2);

        ctx.fillStyle = changeColor;
        ctx.fillText(`$${point.c.toFixed(2)}`, tooltipX + 28, ohlcY + lineHeight * 3);

        // Change indicator on the right
        ctx.fillStyle = changeColor;
        ctx.textAlign = 'right';
        ctx.fillText(`${changeSign}${changePercent.toFixed(2)}%`, tooltipX + tooltipWidth - 10, ohlcY + lineHeight * 3);
    } else {
        // Simple display for line chart
        ctx.fillStyle = colors.textStrong;
        ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(`$${point.c.toFixed(2)}`, tooltipX + 10, tooltipY + 26);

        ctx.fillStyle = changeColor;
        ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.fillText(`${changeSign}${change.toFixed(2)} (${changeSign}${changePercent.toFixed(2)}%)`, tooltipX + 10, tooltipY + 50);
    }
}

function handleChartMouseMove(e) {
    const { data, canvas, padding } = chartState;
    if (!data) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const chartWidth = width - padding.left - padding.right;
    const mouseX = e.clientX - rect.left;

    const relativeX = mouseX - padding.left;
    const index = Math.round((relativeX / chartWidth) * (data.length - 1));

    if (index >= 0 && index < data.length && index !== chartState.hoveredIndex) {
        chartState.hoveredIndex = index;
        drawChart(1);
    }
}

function handleChartMouseLeave() {
    chartState.hoveredIndex = -1;
    drawChart(1);
}

async function loadFinancials(ticker) {
    const cacheKey = `financials_${ticker}`;
    const container = document.getElementById('financialsData');
    container.innerHTML = '<p>Loading financial data...</p>';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderFinancials(data, container);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/financials`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.MODERATE);

        renderFinancials(data, container);
    } catch (error) {
        console.error('Error loading financials:', error);
        container.innerHTML = '<p>Error loading financial data.</p>';
    }
}

function renderFinancials(data, container) {
    if (data.results && data.results.length > 0) {
        let html = '';

        data.results.forEach(period => {
            const financials = period.financials;
            const endDate = period.end_date ? new Date(period.end_date).toLocaleDateString() : '';
            const dateDisplay = endDate ? ` (${endDate})` : '';

            html += `
                <div class="financial-period">
                    <h4>${period.fiscal_year} - ${period.fiscal_period}${dateDisplay}</h4>
                    <div class="financial-grid">
            `;

            if (financials.income_statement) {
                const income = financials.income_statement;
                if (income.revenues) {
                    html += `
                        <div class="financial-item">
                            <span class="financial-item-label">Revenue</span>
                            <span class="financial-item-value">${formatLargeNumber(income.revenues.value)}</span>
                        </div>
                    `;
                }
                if (income.net_income_loss) {
                    html += `
                        <div class="financial-item">
                            <span class="financial-item-label">Net Income</span>
                            <span class="financial-item-value">${formatLargeNumber(income.net_income_loss.value)}</span>
                        </div>
                    `;
                }
                if (income.gross_profit) {
                    html += `
                        <div class="financial-item">
                            <span class="financial-item-label">Gross Profit</span>
                            <span class="financial-item-value">${formatLargeNumber(income.gross_profit.value)}</span>
                        </div>
                    `;
                }
            }

            if (financials.balance_sheet) {
                const balance = financials.balance_sheet;
                if (balance.assets) {
                    html += `
                        <div class="financial-item">
                            <span class="financial-item-label">Total Assets</span>
                            <span class="financial-item-value">${formatLargeNumber(balance.assets.value)}</span>
                        </div>
                    `;
                }
                if (balance.liabilities) {
                    html += `
                        <div class="financial-item">
                            <span class="financial-item-label">Total Liabilities</span>
                            <span class="financial-item-value">${formatLargeNumber(balance.liabilities.value)}</span>
                        </div>
                    `;
                }
            }

            html += `
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } else {
        container.innerHTML = '<p>No financial data available.</p>';
    }
}

async function loadNews(ticker) {
    const cacheKey = `news_${ticker}`;
    const container = document.getElementById('newsContainer');
    container.innerHTML = '<p>Loading news...</p>';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderNews(data, container);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/news?limit=10`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.SHORT);

        renderNews(data, container);

        // Trigger article scraping in background
        scrapeAndEmbedArticles();
    } catch (error) {
        console.error('Error loading news:', error);
        container.innerHTML = '<p>Error loading news.</p>';
    }
}

function renderNews(data, container) {
    if (data.results && data.results.length > 0) {
        let html = '';

        data.results.forEach(article => {
            const date = new Date(article.published_utc).toLocaleDateString();
            html += `
                <div class="news-article">
                    <h4><a href="${article.article_url}" target="_blank">${article.title}</a></h4>
                    <div class="news-meta">
                        ${article.publisher?.name || 'Unknown'} - ${date}
                    </div>
                    <div class="news-description">
                        ${article.description || ''}
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    } else {
        container.innerHTML = '<p>No news available.</p>';
    }
}

function formatLargeNumber(num) {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.remove('hidden');
    } else {
        loading.classList.add('hidden');
    }
}

async function loadMarketStatus() {
    const cacheKey = 'market_status';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderMarketStatus(data);
        return;
    }

    try {
        const response = await fetch('http://localhost:5000/api/market-status');
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.DAILY);

        renderMarketStatus(data);
    } catch (error) {
        console.error('Error loading market status:', error);
        document.getElementById('marketStatusText').textContent = 'Unknown';
    }
}

function renderMarketStatus(data) {
    const statusText = document.getElementById('marketStatusText');
    if (data.market === 'open') {
        statusText.textContent = 'Open';
        statusText.classList.add('status-open');
        statusText.classList.remove('status-closed');
    } else {
        statusText.textContent = 'Closed';
        statusText.classList.add('status-closed');
        statusText.classList.remove('status-open');
    }
}

async function loadDividends(ticker) {
    const cacheKey = `dividends_${ticker}`;
    const container = document.getElementById('dividendsContainer');
    container.innerHTML = '<p>Loading dividend data...</p>';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderDividends(data, container);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/dividends?limit=20`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.STATIC);

        renderDividends(data, container);
    } catch (error) {
        console.error('Error loading dividends:', error);
        container.innerHTML = '<p>Error loading dividend data.</p>';
    }
}

function renderDividends(data, container) {
    if (data.results && data.results.length > 0) {
        let html = '<table class="data-table"><thead><tr>';
        html += '<th>Ex-Dividend Date</th>';
        html += '<th>Pay Date</th>';
        html += '<th>Amount</th>';
        html += '<th>Frequency</th>';
        html += '</tr></thead><tbody>';

        data.results.forEach(dividend => {
            const exDate = new Date(dividend.ex_dividend_date).toLocaleDateString();
            const payDate = dividend.pay_date ? new Date(dividend.pay_date).toLocaleDateString() : 'N/A';
            const frequency = getFrequencyText(dividend.frequency);

            html += '<tr>';
            html += `<td>${exDate}</td>`;
            html += `<td>${payDate}</td>`;
            html += `<td>$${dividend.cash_amount.toFixed(4)}</td>`;
            html += `<td>${frequency}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<p>No dividend data available for this stock.</p>';
    }
}

async function loadSplits(ticker) {
    const cacheKey = `splits_${ticker}`;
    const container = document.getElementById('splitsContainer');
    container.innerHTML = '<p>Loading stock split data...</p>';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderSplits(data, container);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/splits?limit=20`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.STATIC);

        renderSplits(data, container);
    } catch (error) {
        console.error('Error loading splits:', error);
        container.innerHTML = '<p>Error loading stock split data.</p>';
    }
}

function renderSplits(data, container) {
    if (data.results && data.results.length > 0) {
        let html = '<table class="data-table"><thead><tr>';
        html += '<th>Execution Date</th>';
        html += '<th>Split Ratio</th>';
        html += '<th>Type</th>';
        html += '</tr></thead><tbody>';

        data.results.forEach(split => {
            const execDate = new Date(split.execution_date).toLocaleDateString();
            const ratio = `${split.split_to}:${split.split_from}`;
            const type = split.split_from > split.split_to ? 'Reverse Split' : 'Forward Split';

            html += '<tr>';
            html += `<td>${execDate}</td>`;
            html += `<td>${ratio}</td>`;
            html += `<td>${type}</td>`;
            html += '</tr>';
        });

        html += '</tbody></table>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<p>No stock split data available for this stock.</p>';
    }
}

function getFrequencyText(frequency) {
    const frequencies = {
        0: 'One-time',
        1: 'Annual',
        2: 'Semi-Annual',
        4: 'Quarterly',
        12: 'Monthly',
        24: 'Bi-Monthly',
        52: 'Weekly'
    };
    return frequencies[frequency] || 'Unknown';
}

// ============================================
// CHAT FUNCTIONALITY
// ============================================

// Chat state management
let chatState = {
    conversationId: generateUUID(),
    messages: [],
    isOpen: true,  // Default to open for persistent panel
    isLoading: false
};

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function updateChatContext() {
    const contextEl = document.getElementById('chatCurrentTicker');

    if (currentTicker) {
        const detailsCache = cache.get(`details_${currentTicker}`);
        const companyName = detailsCache?.results?.name || currentTicker;
        contextEl.textContent = `${currentTicker} - ${companyName}`;
    } else {
        contextEl.textContent = 'Select a stock to start';
    }
}

// Tool display names for status indicators
const TOOL_DISPLAY_NAMES = {
    'get_stock_quote': 'Fetching stock price',
    'get_company_info': 'Looking up company info',
    'get_financials': 'Retrieving financial data',
    'get_news': 'Searching news articles',
    'search_knowledge_base': 'Searching knowledge base',
    'analyze_sentiment': 'Analyzing social sentiment',
    'get_price_forecast': 'Generating price forecast',
    'get_dividends': 'Fetching dividend history',
    'get_stock_splits': 'Checking split history',
    'get_price_history': 'Loading price history'
};

function parseSSEBuffer(buffer) {
    const parsed = [];
    const lines = buffer.split('\n');
    let currentEvent = { type: null, data: null };
    let remaining = '';

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        if (line.startsWith('event: ')) {
            currentEvent.type = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
            currentEvent.data = line.substring(6);
        } else if (line === '' && currentEvent.type !== null) {
            let data = currentEvent.data;
            if (currentEvent.type !== 'text') {
                try { data = JSON.parse(data); } catch (e) {}
            }
            parsed.push({ type: currentEvent.type, data: data });
            currentEvent = { type: null, data: null };
        }
    }

    // Keep incomplete event in buffer
    if (currentEvent.type !== null || currentEvent.data !== null) {
        remaining = '';
        if (currentEvent.type) remaining += `event: ${currentEvent.type}\n`;
        if (currentEvent.data !== null) remaining += `data: ${currentEvent.data}\n`;
    }

    return { parsed, remaining };
}

function renderToolStatuses(elementId, statuses) {
    const el = document.getElementById(elementId);
    if (!el) return;

    let html = '';
    for (const status of statuses) {
        const icon = status.status === 'complete' ? '&#10003;' :
                     status.status === 'error' ? '&#10007;' :
                     '<span class="tool-spinner"></span>';
        const className = `tool-status-item ${status.status}`;
        html += `<div class="${className}">${icon} ${escapeHtml(status.displayName)}</div>`;
    }
    el.innerHTML = html;
    el.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();

    if (!message) return;

    if (!currentTicker) {
        addMessageToChat('error', 'Please select a stock first.');
        return;
    }

    // Add user message to UI
    addMessageToChat('user', message);
    input.value = '';

    // Disable input while processing
    input.disabled = true;
    document.getElementById('sendChatBtn').disabled = true;

    // Show loading indicator
    const loadingId = addMessageToChat('loading', '');

    // Collect current stock context from cache (for agent's hybrid caching)
    const context = {
        overview: {
            details: cache.get(`details_${currentTicker}`),
            previousClose: cache.get(`prev_close_${currentTicker}`)
        },
        financials: cache.get(`financials_${currentTicker}`),
        news: cache.get(`news_${currentTicker}`),
        dividends: cache.get(`dividends_${currentTicker}`),
        splits: cache.get(`splits_${currentTicker}`),
        sentiment: cache.get(`sentiment_${currentTicker}`)
    };

    try {
        const response = await fetch(`${API_BASE}/chat/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: currentTicker,
                message: message,
                context: context,
                conversation_id: chatState.conversationId
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Remove loading indicator
        removeMessage(loadingId);

        // Create tool status container and assistant message container
        const toolStatusId = addMessageToChat('tool-status', '');
        let toolStatuses = [];
        const messageId = addMessageToChat('assistant', '');
        let assistantMessage = '';

        // Parse structured SSE events
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const result = parseSSEBuffer(buffer);
            buffer = result.remaining;

            for (const event of result.parsed) {
                if (event.type === 'tool_call') {
                    const displayName = TOOL_DISPLAY_NAMES[event.data.tool] || event.data.tool;

                    if (event.data.status === 'calling') {
                        toolStatuses.push({ tool: event.data.tool, displayName, status: 'calling' });
                    } else {
                        const existing = toolStatuses.find(t => t.tool === event.data.tool);
                        if (existing) existing.status = event.data.status;
                    }
                    renderToolStatuses(toolStatusId, toolStatuses);

                } else if (event.type === 'text') {
                    assistantMessage += event.data;
                    updateMessage(messageId, assistantMessage);

                } else if (event.type === 'done') {
                    // Fade out tool statuses
                    const toolEl = document.getElementById(toolStatusId);
                    if (toolEl && toolStatuses.length > 0) {
                        toolEl.classList.add('tool-status-complete');
                    }

                } else if (event.type === 'error') {
                    removeMessage(messageId);
                    addMessageToChat('error', event.data.message || 'An error occurred');
                }
            }
        }

        // Remove tool status container if no tools were called
        if (toolStatuses.length === 0) {
            removeMessage(toolStatusId);
        }

        // Save to chat state
        chatState.messages.push(
            { role: 'user', content: message },
            { role: 'assistant', content: assistantMessage }
        );

    } catch (error) {
        console.error('Chat error:', error);
        removeMessage(loadingId);
        addMessageToChat('error', 'Failed to get response. Please try again.');
    } finally {
        input.disabled = false;
        document.getElementById('sendChatBtn').disabled = false;
        input.focus();
    }
}

function addMessageToChat(type, content) {
    const container = document.getElementById('chatMessages');
    const messageId = generateUUID();

    const messageDiv = document.createElement('div');
    messageDiv.id = messageId;
    messageDiv.className = `message ${type}`;

    if (type === 'loading') {
        messageDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    } else {
        messageDiv.textContent = content;
    }

    container.appendChild(messageDiv);
    messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

    return messageId;
}

function updateMessage(messageId, content) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        messageDiv.textContent = content;
        messageDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
}

function removeMessage(messageId) {
    const messageDiv = document.getElementById(messageId);
    if (messageDiv) {
        messageDiv.remove();
    }
}

// Preload news data and trigger RAG scraping when stock is selected
async function preloadNewsForRAG(ticker) {
    const cacheKey = `news_${ticker}`;

    // Skip if already cached
    if (cache.has(cacheKey)) {
        // Trigger scraping with cached data
        scrapeAndEmbedArticles();
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/news?limit=10`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.SHORT);

        // Trigger article scraping in background
        scrapeAndEmbedArticles();
    } catch (error) {
        console.error('Error preloading news for RAG:', error);
    }
}

// Background job to scrape and embed articles when News tab is loaded
async function scrapeAndEmbedArticles() {
    if (!currentTicker) return;

    const newsCache = cache.get(`news_${currentTicker}`);
    if (!newsCache || !newsCache.results) return;

    try {
        // Call scrape endpoint in background (don't await)
        fetch(`${API_BASE}/chat/scrape-articles`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: currentTicker,
                articles: newsCache.results
            })
        }).then(response => response.json())
          .then(result => {
              console.log('Article scraping complete:', result);
          })
          .catch(error => {
              console.error('Article scraping error:', error);
          });
    } catch (error) {
        console.error('Failed to initiate article scraping:', error);
    }
}

// Setup chat event listeners
function setupChatListeners() {
    // Mobile chat toggle button
    const mobileToggle = document.getElementById('mobileChatToggle');
    if (mobileToggle) {
        mobileToggle.addEventListener('click', toggleMobileChat);
    }

    // Mobile chat close button
    const mobileClose = document.getElementById('closeMobileChat');
    if (mobileClose) {
        mobileClose.addEventListener('click', toggleMobileChat);
    }

    // Send button and input
    document.getElementById('sendChatBtn').addEventListener('click', sendChatMessage);

    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });

    // Setup sentiment listeners
    setupSentimentListeners();

    // Setup forecast listeners
    setupForecastListeners();
}

// Toggle chat for mobile (full screen overlay)
function toggleMobileChat() {
    const chatPanel = document.getElementById('chatPanel');
    const isOpen = chatPanel.classList.contains('open');

    if (isOpen) {
        chatPanel.classList.remove('open');
    } else {
        chatPanel.classList.add('open');
        updateChatContext();
    }
}

// ============================================
// SENTIMENT ANALYSIS FUNCTIONALITY
// ============================================

let sentimentState = {
    currentFilter: 'all',
    posts: [],
    isLoading: false
};

function setupSentimentListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshSentiment');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            if (currentTicker) {
                // Clear cache and reload
                const cacheKey = `sentiment_${currentTicker}`;
                delete cache.data[cacheKey];
                loadSentiment(currentTicker, true);
            }
        });
    }

    // Filter buttons
    const filterBtns = document.querySelectorAll('.posts-filter .filter-btn');
    filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            sentimentState.currentFilter = btn.dataset.filter;
            renderSentimentPosts(sentimentState.posts);
        });
    });
}

async function loadSentiment(ticker, forceRefresh = false) {
    const cacheKey = `sentiment_${ticker}`;
    const container = document.getElementById('sentimentPostsContainer');

    // Show loading state
    if (!forceRefresh) {
        container.innerHTML = '<p class="loading-text">Analyzing social media sentiment...</p>';
    }

    // Update refresh button state
    const refreshBtn = document.getElementById('refreshSentiment');
    if (refreshBtn) {
        refreshBtn.disabled = true;
        refreshBtn.classList.add('loading');
    }

    sentimentState.isLoading = true;

    // Check cache first (unless forcing refresh)
    if (!forceRefresh && cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderSentiment(data);
        sentimentState.isLoading = false;
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.classList.remove('loading');
        }
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/sentiment/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Cache with short TTL (15 minutes)
        cache.set(cacheKey, data, CACHE_TTL.SHORT);

        renderSentiment(data);

    } catch (error) {
        console.error('Error loading sentiment:', error);
        container.innerHTML = '<p class="error-text">Error loading sentiment data. Please try again.</p>';
        resetSentimentUI();
    } finally {
        sentimentState.isLoading = false;
        if (refreshBtn) {
            refreshBtn.disabled = false;
            refreshBtn.classList.remove('loading');
        }
    }
}

function renderSentiment(data) {
    const aggregate = data.aggregate;

    // Update gauge
    updateSentimentGauge(aggregate.score);

    // Update label and confidence
    const labelEl = document.getElementById('sentimentLabel');
    labelEl.textContent = aggregate.label.toUpperCase();
    labelEl.className = 'sentiment-label ' + aggregate.label;

    // Update stats
    document.getElementById('sentimentPostCount').textContent = aggregate.post_count;
    document.getElementById('sentimentLastUpdated').textContent = new Date().toLocaleTimeString();

    // Update source breakdown
    const sources = aggregate.sources || {};
    updateSourceItem('stocktwitsSource', sources.stocktwits || 0, aggregate.post_count);
    updateSourceItem('redditSource', sources.reddit || 0, aggregate.post_count);
    updateSourceItem('twitterSource', sources.twitter || 0, aggregate.post_count);

    // Store posts and render
    sentimentState.posts = data.posts || [];
    renderSentimentPosts(sentimentState.posts);
}

function updateSentimentGauge(score) {
    // Score ranges from -1 (bearish) to +1 (bullish)
    // Map to rotation: -90deg (bearish) to +90deg (bullish)
    const rotation = score * 90;

    const needle = document.getElementById('gaugeNeedle');
    if (needle) {
        needle.style.transform = `rotate(${rotation}deg)`;
    }

    // Update gauge fill color based on sentiment
    const fill = document.getElementById('gaugeFill');
    if (fill) {
        if (score > 0.2) {
            fill.className = 'gauge-fill bullish';
        } else if (score < -0.2) {
            fill.className = 'gauge-fill bearish';
        } else {
            fill.className = 'gauge-fill neutral';
        }
    }
}

function updateSourceItem(elementId, count, total) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const countEl = element.querySelector('.source-count');
    if (countEl) {
        countEl.textContent = count;
    }
}

function renderSentimentPosts(posts) {
    const container = document.getElementById('sentimentPostsContainer');

    if (!posts || posts.length === 0) {
        container.innerHTML = '<p class="no-data-text">No sentiment data available for this stock.</p>';
        return;
    }

    // Apply filter
    let filteredPosts = posts;
    if (sentimentState.currentFilter !== 'all') {
        filteredPosts = posts.filter(post => {
            const label = post.sentiment?.label || 'neutral';
            return label === sentimentState.currentFilter;
        });
    }

    if (filteredPosts.length === 0) {
        container.innerHTML = `<p class="no-data-text">No ${sentimentState.currentFilter} posts found.</p>`;
        return;
    }

    let html = '';
    filteredPosts.forEach(post => {
        const sentimentLabel = post.sentiment?.label || 'neutral';
        const sentimentScore = post.sentiment?.score || 0;
        const scorePercent = (sentimentScore * 100).toFixed(0);
        const timestamp = post.timestamp ? formatRelativeTime(post.timestamp) : '';
        const platform = post.platform || 'unknown';
        const engagement = post.engagement || {};

        html += `
            <div class="sentiment-post ${sentimentLabel}">
                <div class="post-header">
                    <span class="post-platform">${getPlatformIcon(platform)} ${platform}</span>
                    <span class="post-sentiment ${sentimentLabel}">
                        ${sentimentLabel} (${scorePercent}%)
                    </span>
                </div>
                <p class="post-content">${escapeHtml(post.content || '')}</p>
                <div class="post-meta">
                    <span class="post-author">@${escapeHtml(post.author || 'unknown')}</span>
                    <span class="post-time">${timestamp}</span>
                    <span class="post-engagement">
                        ${engagement.likes || 0} likes ${engagement.comments ? `· ${engagement.comments} comments` : ''}
                    </span>
                </div>
                ${post.url ? `<a href="${post.url}" target="_blank" class="post-link">View original</a>` : ''}
            </div>
        `;
    });

    container.innerHTML = html;
}

function getPlatformIcon(platform) {
    const icons = {
        'stocktwits': '📈',
        'reddit': '👽',
        'twitter': '🐦'
    };
    return icons[platform] || '💬';
}

function formatRelativeTime(timestamp) {
    try {
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);

        if (diffMins < 1) return 'just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        if (diffDays < 7) return `${diffDays}d ago`;
        return date.toLocaleDateString();
    } catch {
        return '';
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function resetSentimentUI() {
    document.getElementById('sentimentLabel').textContent = '--';
    document.getElementById('sentimentLabel').className = 'sentiment-label';
    document.getElementById('sentimentConfidence').textContent = '--';
    document.getElementById('sentimentPostCount').textContent = '--';
    document.getElementById('sentimentLastUpdated').textContent = '--';

    updateSourceItem('stocktwitsSource', 0, 0);
    updateSourceItem('redditSource', 0, 0);
    updateSourceItem('twitterSource', 0, 0);

    const needle = document.getElementById('gaugeNeedle');
    if (needle) {
        needle.style.transform = 'rotate(0deg)';
    }
}

// ============================================
// FORECAST FUNCTIONALITY
// ============================================

let forecastState = {
    data: null,
    isLoading: false,
    modelStatus: null
};

let forecastChartState = {
    data: null,
    canvas: null,
    ctx: null,
    padding: { top: 20, right: 20, bottom: 40, left: 65 },
    hoveredIndex: -1,
    totalPoints: 0,
    historicalLength: 0
};

function setupForecastListeners() {
}

async function getHistoricalDataForForecast(ticker) {
    // Try to get 2-year data from cache first (used for training)
    const cacheKey2Y = `chart_${ticker}_2Y`;
    if (cache.has(cacheKey2Y)) {
        const data = cache.get(cacheKey2Y);
        if (data.results && data.results.length > 0) {
            return data.results;
        }
    }

    // Try 5-year cache as fallback (has more than enough data)
    const cacheKey5Y = `chart_${ticker}_5Y`;
    if (cache.has(cacheKey5Y)) {
        const data = cache.get(cacheKey5Y);
        if (data.results && data.results.length > 0) {
            return data.results;
        }
    }

    // Try 1-year cache (minimum for decent training)
    const cacheKey1Y = `chart_${ticker}_1Y`;
    if (cache.has(cacheKey1Y)) {
        const data = cache.get(cacheKey1Y);
        if (data.results && data.results.length > 0) {
            return data.results;
        }
    }

    // No cached data available, fetch 2 years of data
    const to = new Date();
    const from = new Date();
    from.setFullYear(from.getFullYear() - 2);

    const fromStr = from.toISOString().split('T')[0];
    const toStr = to.toISOString().split('T')[0];

    try {
        const response = await fetch(
            `${API_BASE}/ticker/${ticker}/aggregates?from=${fromStr}&to=${toStr}&timespan=day`
        );
        const data = await response.json();

        // Cache it for future use
        if (data.results) {
            cache.set(cacheKey2Y, data, CACHE_TTL.DAILY);
            return data.results;
        }
    } catch (error) {
        console.error('Error fetching historical data for forecast:', error);
    }

    return null;
}

async function loadForecast(ticker) {
    const cacheKey = `forecast_${ticker}`;
    const container = document.getElementById('forecastTableContainer');

    // Show loading state
    container.innerHTML = '<p class="loading-text">Loading forecast...</p>';

    forecastState.isLoading = true;

    // Update status to show loading
    document.getElementById('forecastModelStatus').textContent = 'Loading...';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderForecast(data);
        forecastState.isLoading = false;
        return;
    }

    try {
        // Get historical data from cache or fetch it (reuse existing chart data)
        const historicalData = await getHistoricalDataForForecast(ticker);

        const response = await fetch(`${API_BASE}/forecast/predict/${ticker}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                historical_data: historicalData
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Forecast failed');
        }

        const data = await response.json();
        cache.set(cacheKey, data, CACHE_TTL.MODERATE);
        renderForecast(data);

    } catch (error) {
        console.error('Error loading forecast:', error);
        container.innerHTML = `<p class="error-text">${error.message || 'Error loading forecast'}</p>`;
        resetForecastUI();
    } finally {
        forecastState.isLoading = false;
    
    }
}

function renderForecast(data) {
    forecastState.data = data;

    // Clear loading text
    document.getElementById('forecastTableContainer').innerHTML = '';

    // Update status
    if (data.model_info) {
        document.getElementById('forecastModelStatus').textContent = 'Trained';
        const trainedAt = data.model_info.trained_at;
        if (trainedAt) {
            const date = new Date(trainedAt);
            document.getElementById('forecastLastUpdated').textContent = date.toLocaleDateString();
        }
    } else {
        document.getElementById('forecastModelStatus').textContent = 'Not trained';
    }

    // Draw forecast chart
    drawForecastChart(data);

}

function drawForecastChart(data, skipSetup = false) {
    const canvas = document.getElementById('forecastChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const colors = getChartColors();

    // High DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    if (!skipSetup) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
    }

    const padding = { top: 20, right: 20, bottom: 40, left: 65 };
    const width = rect.width;
    const height = rect.height;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Combine historical and forecast data
    const historical = data.historical || [];
    const forecast = data.forecast || [];
    const confidenceBounds = data.confidence_bounds || {};

    if (historical.length === 0 && forecast.length === 0) {
        ctx.fillStyle = colors.text;
        ctx.font = '14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No data available', width / 2, height / 2);
        return;
    }

    // Store state for hover interactions
    forecastChartState.data = data;
    forecastChartState.canvas = canvas;
    forecastChartState.ctx = ctx;
    forecastChartState.totalPoints = historical.length + forecast.length;
    forecastChartState.historicalLength = historical.length;

    // Calculate price range
    const historicalPrices = historical.map(d => d.close);
    const forecastPrices = forecast.map(d => d.predicted_close);
    const upperBound = confidenceBounds.upper || [];
    const lowerBound = confidenceBounds.lower || [];

    const allPrices = [...historicalPrices, ...forecastPrices, ...upperBound, ...lowerBound];
    const minPrice = Math.min(...allPrices);
    const maxPrice = Math.max(...allPrices);
    const pricePadding = (maxPrice - minPrice) * 0.1;
    const adjustedMin = minPrice - pricePadding;
    const adjustedMax = maxPrice + pricePadding;
    const priceRange = adjustedMax - adjustedMin;

    const totalPoints = historical.length + forecast.length;

    ctx.clearRect(0, 0, width, height);

    // Draw grid lines
    const numGridLines = 5;
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.fillStyle = colors.text;
    ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let i = 0; i <= numGridLines; i++) {
        const y = padding.top + (i / numGridLines) * chartHeight;
        const price = adjustedMax - (i / numGridLines) * priceRange;

        ctx.beginPath();
        ctx.setLineDash([4, 4]);
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillText(`$${price.toFixed(2)}`, padding.left - 8, y);
    }

    // Draw vertical divider line between historical and forecast
    if (historical.length > 0 && forecast.length > 0) {
        const dividerX = padding.left + (historical.length / totalPoints) * chartWidth;
        ctx.strokeStyle = colors.crosshair;
        ctx.lineWidth = 1;
        ctx.setLineDash([8, 4]);
        ctx.beginPath();
        ctx.moveTo(dividerX, padding.top);
        ctx.lineTo(dividerX, height - padding.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw confidence band (shaded area)
    if (upperBound.length > 0 && lowerBound.length > 0) {
        ctx.beginPath();
        ctx.fillStyle = 'rgba(16, 185, 129, 0.15)';

        // Start from first forecast point
        const startIdx = historical.length;
        for (let i = 0; i < forecast.length; i++) {
            const x = padding.left + ((startIdx + i) / (totalPoints - 1)) * chartWidth;
            const y = padding.top + ((adjustedMax - upperBound[i]) / priceRange) * chartHeight;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }

        // Go back along lower bound
        for (let i = forecast.length - 1; i >= 0; i--) {
            const x = padding.left + ((startIdx + i) / (totalPoints - 1)) * chartWidth;
            const y = padding.top + ((adjustedMax - lowerBound[i]) / priceRange) * chartHeight;
            ctx.lineTo(x, y);
        }

        ctx.closePath();
        ctx.fill();
    }

    // Draw historical line
    if (historical.length > 1) {
        ctx.beginPath();
        ctx.strokeStyle = colors.line;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        historical.forEach((d, i) => {
            const x = padding.left + (i / (totalPoints - 1)) * chartWidth;
            const y = padding.top + ((adjustedMax - d.close) / priceRange) * chartHeight;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // Draw forecast line (dashed)
    if (forecast.length > 0) {
        ctx.beginPath();
        ctx.strokeStyle = colors.positive;
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // Connect from last historical point
        if (historical.length > 0) {
            const lastHistorical = historical[historical.length - 1];
            const x = padding.left + ((historical.length - 1) / (totalPoints - 1)) * chartWidth;
            const y = padding.top + ((adjustedMax - lastHistorical.close) / priceRange) * chartHeight;
            ctx.moveTo(x, y);
        }

        forecast.forEach((d, i) => {
            const x = padding.left + ((historical.length + i) / (totalPoints - 1)) * chartWidth;
            const y = padding.top + ((adjustedMax - d.predicted_close) / priceRange) * chartHeight;
            if (historical.length === 0 && i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
    }

    // Draw X-axis labels
    ctx.fillStyle = colors.text;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    // Show a few date labels
    const allDates = [...historical.map(d => d.date), ...forecast.map(d => d.date)];
    const labelIndices = [0, Math.floor(historical.length / 2), historical.length - 1, historical.length + Math.floor(forecast.length / 2), totalPoints - 1];

    labelIndices.forEach(idx => {
        if (idx >= 0 && idx < allDates.length) {
            const x = padding.left + (idx / (totalPoints - 1)) * chartWidth;
            const date = new Date(allDates[idx]);
            const label = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            ctx.fillText(label, x, height - padding.bottom + 8);
        }
    });

    // Draw crosshair and tooltip if hovering
    if (forecastChartState.hoveredIndex >= 0 && forecastChartState.hoveredIndex < totalPoints) {
        drawForecastCrosshair(forecastChartState.hoveredIndex, data, adjustedMin, adjustedMax, priceRange, chartWidth, chartHeight, width, height, padding, colors);
    }

    // Set up mouse events (only on initial draw)
    if (!skipSetup) {
        canvas.onmousemove = handleForecastChartMouseMove;
        canvas.onmouseleave = handleForecastChartMouseLeave;
    }
}

function drawForecastCrosshair(index, data, adjustedMin, adjustedMax, priceRange, chartWidth, chartHeight, width, height, padding, colors) {
    const ctx = forecastChartState.ctx;
    const historical = data.historical || [];
    const forecast = data.forecast || [];
    const totalPoints = historical.length + forecast.length;

    // Determine if we're in historical or forecast region
    const isHistorical = index < historical.length;
    let point, price, date;

    if (isHistorical) {
        point = historical[index];
        price = point.close;
        date = point.date;
    } else {
        const forecastIdx = index - historical.length;
        point = forecast[forecastIdx];
        price = point.predicted_close;
        date = point.date;
    }

    const x = padding.left + (index / (totalPoints - 1)) * chartWidth;
    const y = padding.top + ((adjustedMax - price) / priceRange) * chartHeight;

    // Crosshair lines
    ctx.strokeStyle = colors.crosshair;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Vertical line
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, height - padding.bottom);
    ctx.stroke();

    // Horizontal line
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Point dot
    ctx.beginPath();
    ctx.fillStyle = isHistorical ? colors.line : colors.positive;
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = colors.tooltipBg;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Tooltip
    drawForecastTooltip(x, y, point, isHistorical, width, height, padding, colors);
}

function drawForecastTooltip(x, y, point, isHistorical, width, height, padding, colors) {
    const ctx = forecastChartState.ctx;

    const date = new Date(point.date);
    const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });

    const price = isHistorical ? point.close : point.predicted_close;
    const tooltipWidth = isHistorical ? 140 : 155;
    const tooltipHeight = isHistorical ? 52 : 72;
    let tooltipX = x + 12;
    let tooltipY = y - tooltipHeight / 2;

    // Keep tooltip in bounds
    if (tooltipX + tooltipWidth > width - padding.right) {
        tooltipX = x - tooltipWidth - 12;
    }
    if (tooltipY < padding.top) {
        tooltipY = padding.top;
    }
    if (tooltipY + tooltipHeight > height - padding.bottom) {
        tooltipY = height - padding.bottom - tooltipHeight;
    }

    // Tooltip background
    ctx.fillStyle = colors.tooltipBg;
    ctx.strokeStyle = colors.tooltipBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(tooltipX, tooltipY, tooltipWidth, tooltipHeight, 6);
    ctx.fill();
    ctx.stroke();

    // Tooltip content
    ctx.fillStyle = colors.text;
    ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(dateStr, tooltipX + 10, tooltipY + 8);

    ctx.fillStyle = colors.textStrong;
    ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
    ctx.fillText(`$${price.toFixed(2)}`, tooltipX + 10, tooltipY + 24);

    if (!isHistorical) {
        // Show confidence range for predictions
        ctx.fillStyle = colors.text;
        ctx.font = '10px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
        const rangeStr = `Range: $${point.lower_bound.toFixed(2)} - $${point.upper_bound.toFixed(2)}`;
        ctx.fillText(rangeStr, tooltipX + 10, tooltipY + 48);
    }
}

function handleForecastChartMouseMove(e) {
    const { data, canvas, padding, totalPoints } = forecastChartState;
    if (!data) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.width / dpr;
    const chartWidth = width - padding.left - padding.right;
    const mouseX = e.clientX - rect.left;

    const relativeX = mouseX - padding.left;
    const index = Math.round((relativeX / chartWidth) * (totalPoints - 1));

    if (index >= 0 && index < totalPoints && index !== forecastChartState.hoveredIndex) {
        forecastChartState.hoveredIndex = index;
        drawForecastChart(data, true);
    }
}

function handleForecastChartMouseLeave() {
    forecastChartState.hoveredIndex = -1;
    if (forecastChartState.data) {
        drawForecastChart(forecastChartState.data, true);
    }
}



function resetForecastUI() {
    document.getElementById('forecastModelStatus').textContent = 'Not trained';
    document.getElementById('forecastLastUpdated').textContent = '--';
    forecastState.data = null;

    const canvas = document.getElementById('forecastChart');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}
