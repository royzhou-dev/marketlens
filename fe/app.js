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
    await loadTickers();
    await loadMarketStatus();
    setupEventListeners();
});

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

    if (tabName === 'financials' && currentTicker) {
        loadFinancials(currentTicker);
    } else if (tabName === 'news' && currentTicker) {
        loadNews(currentTicker);
    } else if (tabName === 'dividends' && currentTicker) {
        loadDividends(currentTicker);
    } else if (tabName === 'splits' && currentTicker) {
        loadSplits(currentTicker);
    } else if (tabName === 'related' && currentTicker) {
        loadRelatedCompanies(currentTicker);
    }
}

async function loadStockData(ticker) {
    currentTicker = ticker;
    document.getElementById('stockData').classList.remove('hidden');
    showLoading(true);

    try {
        await Promise.all([
            loadTickerDetails(ticker),
            loadPreviousClose(ticker),
            loadChartData(ticker, '1M')
        ]);

        const activeTab = document.querySelector('.tab-button.active').dataset.tab;
        if (activeTab === 'financials') {
            await loadFinancials(ticker);
        } else if (activeTab === 'news') {
            await loadNews(ticker);
        }
    } catch (error) {
        console.error('Error loading stock data:', error);
        alert('Error loading stock data. Please check your API key and try again.');
    } finally {
        showLoading(false);
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

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.STATIC);

        renderTickerDetails(data);
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

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.DAILY);

        renderPreviousClose(data);
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

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        if (data.results) {
            renderChart(data.results);
        }
        return;
    }

    const { from, to } = getDateRange(range);

    try {
        const response = await fetch(
            `${API_BASE}/ticker/${ticker}/aggregates?from=${from}&to=${to}&timespan=day`
        );
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.DAILY);

        if (data.results) {
            renderChart(data.results);
        }
    } catch (error) {
        console.error('Error loading chart data:', error);
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

function renderChart(data) {
    const canvas = document.getElementById('priceChart');
    const ctx = canvas.getContext('2d');

    canvas.width = canvas.offsetWidth;
    canvas.height = 400;

    const padding = 40;
    const chartWidth = canvas.width - 2 * padding;
    const chartHeight = canvas.height - 2 * padding;

    const prices = data.map(d => d.c);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.beginPath();

    data.forEach((point, index) => {
        const x = padding + (index / (data.length - 1)) * chartWidth;
        const y = canvas.height - padding - ((point.c - minPrice) / priceRange) * chartHeight;

        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });

    ctx.stroke();

    ctx.fillStyle = 'rgba(102, 126, 234, 0.1)';
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.closePath();
    ctx.fill();

    ctx.strokeStyle = '#ddd';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, canvas.height - padding);
    ctx.lineTo(canvas.width - padding, canvas.height - padding);
    ctx.stroke();

    ctx.fillStyle = '#666';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`$${maxPrice.toFixed(2)}`, padding - 5, padding + 5);
    ctx.fillText(`$${minPrice.toFixed(2)}`, padding - 5, canvas.height - padding + 5);
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

async function loadRelatedCompanies(ticker) {
    const cacheKey = `related_${ticker}`;
    const container = document.getElementById('relatedContainer');
    container.innerHTML = '<p>Loading related companies...</p>';

    // Check cache first
    if (cache.has(cacheKey)) {
        const data = cache.get(cacheKey);
        renderRelatedCompanies(data, container);
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/ticker/${ticker}/related`);
        const data = await response.json();

        // Store in cache
        cache.set(cacheKey, data, CACHE_TTL.STATIC);

        renderRelatedCompanies(data, container);
    } catch (error) {
        console.error('Error loading related companies:', error);
        container.innerHTML = '<p>Related companies feature may require a paid API plan.</p>';
    }
}

function renderRelatedCompanies(data, container) {
    if (data.results && data.results.length > 0) {
        let html = '<div class="related-grid">';

        data.results.forEach(company => {
            html += `
                <div class="related-card">
                    <h4>${company.ticker}</h4>
                    <p>${company.name || 'N/A'}</p>
                </div>
            `;
        });

        html += '</div>';
        container.innerHTML = html;
    } else {
        container.innerHTML = '<p>No related companies data available. This feature may require a paid API plan.</p>';
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
    isOpen: false,
    isLoading: false
};

function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

function toggleChatSidebar() {
    const sidebar = document.getElementById('chatSidebar');
    chatState.isOpen = !chatState.isOpen;

    if (chatState.isOpen) {
        sidebar.classList.add('open');
        updateChatContext();
    } else {
        sidebar.classList.remove('open');
    }
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

    // Collect current stock context from cache
    const context = {
        overview: {
            details: cache.get(`details_${currentTicker}`),
            previousClose: cache.get(`prev_close_${currentTicker}`)
        },
        financials: cache.get(`financials_${currentTicker}`),
        news: cache.get(`news_${currentTicker}`),
        dividends: cache.get(`dividends_${currentTicker}`),
        splits: cache.get(`splits_${currentTicker}`)
    };

    try {
        // Call chat API with streaming
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

        // Handle streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let assistantMessage = '';
        const messageId = addMessageToChat('assistant', '');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            assistantMessage += chunk;
            updateMessage(messageId, assistantMessage);
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
        // Re-enable input
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
    document.getElementById('toggleChatBtn').addEventListener('click', toggleChatSidebar);
    document.getElementById('closeChatBtn').addEventListener('click', toggleChatSidebar);
    document.getElementById('sendChatBtn').addEventListener('click', sendChatMessage);

    document.getElementById('chatInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}
