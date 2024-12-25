// Function to fetch and update balance
async function updateBalance() {
    try {
        const response = await fetch('/get_balance');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching balance:', data.error);
            return;
        }
        
        // Update spot balances
        if (data.spot_balance) {
            document.getElementById('usdt-balance').textContent = data.spot_balance.USDT || '0.00';
            document.getElementById('usd-balance').textContent = data.spot_balance.ZUSD || '0.00';
            document.getElementById('btc-balance').textContent = data.spot_balance.XXBT || '0.00';
        }
        
        // Update futures balance if available
        if (data.futures_balance) {
            document.getElementById('kraken-futures-balance').textContent = data.futures_balance;
        }
        
        // Update crypto list search functionality
        const cryptoSearch = document.getElementById('crypto-search');
        if (cryptoSearch) {
            cryptoSearch.addEventListener('input', function() {
                const searchTerm = this.value.toLowerCase();
                const cryptoList = document.getElementById('crypto-list');
                const cryptoItems = cryptoList.getElementsByTagName('li');
                
                for (let item of cryptoItems) {
                    const text = item.textContent.toLowerCase();
                    if (text.includes(searchTerm)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error updating balance:', error);
    }
}

// Function to fetch and update cryptos list
async function fetchCryptos() {
    try {
        const response = await fetch('/get_cryptos');
        const data = await response.json();
        
        if (data.error) {
            console.error('Error fetching cryptos:', data.error);
            return;
        }
        
        const cryptoList = document.getElementById('crypto-list');
        cryptoList.innerHTML = '';
        
        data.cryptos.forEach(crypto => {
            const li = document.createElement('li');
            li.textContent = `${crypto.symbol}: $${parseFloat(crypto.price).toFixed(2)}`;
            li.addEventListener('click', () => {
                updateTradingSymbol(crypto.symbol);
                // Show manual trading controls for this symbol
                document.getElementById('manual-trading-controls').classList.remove('hidden');
                document.getElementById('manual-trade-symbol').textContent = crypto.symbol;
            });
            cryptoList.appendChild(li);
        });
    } catch (error) {
    }
}

// Function to update trading symbol
async function updateTradingSymbol(symbol) {
    try {
        // First get current symbol from backend
        const response = await fetch('/get_current_symbol');
        const data = await response.json();
        
        // Use provided symbol or fallback to current symbol from backend
        const symbolToUse = symbol || data.symbol;
        
        const container = document.getElementById('tradingview_chart');
        container.innerHTML = '';
        window.tradingViewWidget = new TradingView.widget({
            "width": "100%",
            "height": 500,
            "symbol": `KRAKEN:${symbolToUse}`,
            "interval": "1",
            "timezone": "Etc/UTC",
            "theme": "dark", 
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": false,
            "withdateranges": true,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "studies": [
                "RSI@tv-basicstudies",
                "MACD@tv-basicstudies",
                "StochasticRSI@tv-basicstudies"
            ],
            "container_id": "tradingview_chart",
            "autosize": true,
            "hide_top_toolbar": false,
            "save_image": true,
            "show_popup_button": true
        });
        
        // Store current symbol in window object for manual trading
        window.currentTradingSymbol = symbolToUse;
        
    } catch (error) {
        console.error('Error updating symbol:', error);
    }
}

// Function to execute manual trade
async function executeManualTrade(action) {
    try {
        const symbol = document.getElementById('manual-trade-symbol').textContent;
        const sizeType = document.getElementById('manual-trade-size-type').value;
        const amount = parseFloat(document.getElementById('manual-trade-amount').value);
        const orderType = document.getElementById('manual-order-type').value;
        const limitPrice = orderType === 'limit' ? parseFloat(document.getElementById('manual-limit-price').value) : null;
        const takeProfit = parseFloat(document.getElementById('manual-take-profit').value);
        const stopLoss = parseFloat(document.getElementById('manual-stop-loss').value);
        
        if (!amount || amount <= 0) {
            alert('Please enter a valid trade amount');
            return;
        }
        
        if (orderType === 'limit' && (!limitPrice || limitPrice <= 0)) {
            alert('Please enter a valid limit price');
            return;
        }
        
        // Get current balances
        const response = await fetch('/get_balance');
        const balanceData = await response.json();
        
        if (balanceData.error) {
            alert('Error getting balance: ' + balanceData.error);
            return;
        }
        
        const usdtBalance = parseFloat(balanceData.spot_balance?.USDT || 0);
        const cryptoBalance = parseFloat(balanceData.spot_balance?.[symbol.replace('USDT', '')] || 0);
        
        let tradeAmount = amount;
        if (sizeType === 'percentage') {
            if (action === 'buy') {
                tradeAmount = (usdtBalance * amount) / 100;
            } else {
                tradeAmount = (cryptoBalance * amount) / 100;
            }
        }
        
        if (action === 'buy' && tradeAmount > usdtBalance) {
            alert(`Insufficient USDT balance. Available: ${usdtBalance} USDT`);
            return;
        }
        
        if (action === 'sell' && tradeAmount > cryptoBalance) {
            alert(`Insufficient ${symbol.replace('USDT', '')} balance. Available: ${cryptoBalance}`);
            return;
        }
        
        // Execute the trade
        const tradeResponse = await fetch('/execute_trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                action: action,
                volume: tradeAmount,
                ordertype: orderType,
                price: limitPrice,
                take_profit: takeProfit,
                stop_loss: stopLoss
            })
        });
        
        const tradeResult = await tradeResponse.json();
        
        if (tradeResult.error) {
            alert('Trade error: ' + tradeResult.error);
        } else {
            alert(`${action.toUpperCase()} order executed successfully for ${tradeAmount} ${symbol}`);
        }
    } catch (error) {
        console.error('Error executing manual trade:', error);
        alert('Error executing trade. Please try again.');
    }
}

async function handleTradeAction(action, symbol) {
    try {
        const response = await fetch('/execute_trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, symbol })
        });
        const data = await response.json();
        if (data.status === 'success') {
            alert(`Trade executed: ${action} ${symbol}`);
        } else {
            alert(`Error: ${data.message}`);
        }
    } catch (error) {
        console.error('Error executing trade:', error);
        alert('Failed to execute trade. Please try again.');
    }
}

document.addEventListener('DOMContentLoaded', function () {
    // Initialize TradingView widget with only BTC/USDT pair
    let widget = new TradingView.widget({
        "width": "100%",
        "height": 500,
        "symbol": "KRAKEN:XBTUSDT",
        "interval": "15",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "hide_side_toolbar": true,
        "allow_symbol_change": false,
        "studies": [
            "RSI@tv-basicstudies",
            "MACD@tv-basicstudies",
            "StochasticRSI@tv-basicstudies"
        ],
        "container_id": "tradingview_chart",
        "autosize": true,
        "hide_top_toolbar": false,
        "save_image": false,
        "show_popup_button": false,
        "disabled_features": [
            "header_symbol_search",
            "symbol_search_hot_key",
            "header_compare",
            "header_settings",
            "header_screenshot",
            "header_fullscreen_button",
            "compare_symbol",
            "border_around_the_chart",
            "header_undo_redo",
            "show_chart_property_page",
            "symbol_info",
            "symbol_search_hot_key",
            "pane_context_menu",
            "scales_context_menu",
            "legend_context_menu",
            "main_series_scale_menu",
            "display_market_status",
            "remove_library_container_border",
            "create_volume_indicator_by_default",
            "create_volume_indicator_by_default_once",
            "volume_force_overlay"
        ],
        "enabled_features": [],
        "overrides": {
            "mainSeriesProperties.candleStyle.upColor": "#00ff00",
            "mainSeriesProperties.candleStyle.downColor": "#ff0000",
            "mainSeriesProperties.candleStyle.drawWick": true,
            "mainSeriesProperties.candleStyle.drawBorder": true,
            "mainSeriesProperties.candleStyle.borderColor": "#378658",
            "mainSeriesProperties.candleStyle.borderUpColor": "#00ff00",
            "mainSeriesProperties.candleStyle.borderDownColor": "#ff0000",
            "mainSeriesProperties.candleStyle.wickUpColor": "#00ff00",
            "mainSeriesProperties.candleStyle.wickDownColor": "#ff0000"
        },
        "loading_screen": { backgroundColor: "#2d3748" },
        "custom_css_url": "/static/css/chart.css",
        "library_path": "https://s3.tradingview.com/tv.js",
        "fullscreen": false,
        "drawings_access": { type: "black", tools: [] },
        "saved_data": null,
        "auto_save_delay": 0
    });
    // Referral menu functions
    window.showReferralModal = async function() {
        try {
            const response = await fetch('/generate_referral_link');
            const data = await response.json();
            
            if (data.status === 'success') {
                const modalContent = `
                    <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full">
                        <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                            <div class="mt-3 text-center">
                                <h3 class="text-lg font-medium text-gray-900 mb-4">Tu Link de Referido</h3>
                                <div class="flex flex-col space-y-4">
                                    <div class="flex items-center justify-center space-x-2">
                                        <input type="text" value="${data.referral_link}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="referralLinkInput">
                                        <button onclick="copyReferralLink()" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
                                            Copiar
                                        </button>
                                    </div>
                                    <div class="referral-tree">
                                        <h4 class="text-md font-medium text-gray-800 mb-2">Árbol de Referidos</h4>
                                        <div id="referralTreeContainer" class="max-h-60 overflow-y-auto p-2 border rounded">
                                            <ul class="list-none">
                                                <li class="mb-2">
                                                    <span class="font-medium">Tus Referidos Directos:</span>
                                                    <ul class="pl-4 mt-1">
                                                        ${data.direct_referrals ? data.direct_referrals.map(ref => 
                                                            `<li class="text-sm text-gray-600">${ref.name} - ${ref.date}</li>`
                                                        ).join('') : '<li class="text-sm text-gray-500">No hay referidos directos aún</li>'}
                                                    </ul>
                                                </li>
                                                <li>
                                                    <span class="font-medium">Referidos de tus Referidos:</span>
                                                    <ul class="pl-4 mt-1">
                                                        ${data.indirect_referrals ? data.indirect_referrals.map(ref => 
                                                            `<li class="text-sm text-gray-600">${ref.name} - ${ref.date} (Referido por: ${ref.referred_by})</li>`
                                                        ).join('') : '<li class="text-sm text-gray-500">No hay referidos indirectos aún</li>'}
                                                    </ul>
                                                </li>
                                            </ul>
                                        </div>
                                    </div>
                                    <p class="text-sm text-gray-600">Total de referidos: ${data.total_referrals}</p>
                                    <button onclick="closeReferralModal()" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
                                        Cerrar
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>`;
                
                // Add modal to body
                const modalDiv = document.createElement('div');
                modalDiv.id = 'referralLinkModal';
                modalDiv.innerHTML = modalContent;
                document.body.appendChild(modalDiv);
                
                // Add copy function
                window.copyReferralLink = function() {
                    const input = document.getElementById('referralLinkInput');
                    input.select();
                    document.execCommand('copy');
                    alert('Link copiado al portapapeles!');
                };
                
                // Add close function
                window.closeReferralModal = function() {
                    const modal = document.getElementById('referralLinkModal');
                    if (modal) {
                        modal.remove();
                    }
                };
            } else {
                alert('Error generando link de referido: ' + (data.message || 'Error desconocido'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error generando link de referido. Por favor intenta nuevamente.');
        }
        try {
            const response = await fetch('/generate_referral_link');
            const data = await response.json();
            
            if (data.status === 'success') {
                const modalContent = `
                    <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full">
                        <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                            <div class="mt-3 text-center">
                                <h3 class="text-lg font-medium text-gray-900 mb-4">Tu Link de Referido</h3>
                                <div class="flex items-center justify-center space-x-2 mb-4">
                                    <input type="text" value="${data.referral_link}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="referralLinkInput">
                                    <button onclick="copyReferralLink()" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
                                        Copiar
                                    </button>
                                </div>
                                <p class="text-sm text-gray-600 mb-4">Total de referidos: ${data.total_referrals}</p>
                                <button onclick="closeReferralModal()" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
                                    Cerrar
                                </button>
                            </div>
                        </div>
                    </div>`;
                
                // Add modal to body
                const modalDiv = document.createElement('div');
                modalDiv.id = 'referralLinkModal';
                modalDiv.innerHTML = modalContent;
                document.body.appendChild(modalDiv);
                
                // Add copy function
                window.copyReferralLink = function() {
                    const input = document.getElementById('referralLinkInput');
                    input.select();
                    document.execCommand('copy');
                    alert('Link copiado al portapapeles!');
                };
                
                // Add close function
                window.closeReferralModal = function() {
                    const modal = document.getElementById('referralLinkModal');
                    if (modal) {
                        modal.remove();
                    }
                };
            } else {
                alert('Error generando link de referido: ' + (data.message || 'Error desconocido'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error generando link de referido. Por favor intenta nuevamente.');
        }
        toggleReferralMenu();
    };
    
    // Function to generate referral link (legacy)
    window.generateReferralLink = async function() {
        try {
            if (data.status === 'success') {
                alert(`Your referral link: ${data.referral_link}\nTotal referrals: ${data.total_referrals}`);
            } else {
                alert('Error generating referral link: ' + (data.message || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error generating referral link. Please try again.');
        }
    };

    // Initial data fetch
    updateBalance();
    fetchCryptos();
    updateTradingSymbol();

    // Set up periodic updates
    setInterval(updateBalance, 30000); // Update balance every 30 seconds
    setInterval(fetchCryptos, 60000); // Update cryptos every minute
    setInterval(updateTradingSymbol, 60000); // Update symbol every minute

    // Investment modal functionality
    const investmentModal = document.getElementById('investment-modal');
    
    // Function to copy TRON address
    window.copyTronAddress = function() {
        const tronAddress = "TRWzwYKqKvNgZxGgKEyYAqFvWwKFEQkXrk";
        navigator.clipboard.writeText(tronAddress)
            .then(() => alert('Dirección TRON copiada al portapapeles'))
            .catch(() => alert('Error al copiar la dirección'));
    };
    
    // Function to show investment warning
    window.showInvestmentWarning = function() {
        const tronAddress = "TRWzwYKqKvNgZxGgKEyYAqFvWwKFEQkXrk";
        const warningModal = document.createElement('div');
        warningModal.className = 'fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full';
        warningModal.innerHTML = `
            <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                <div class="mt-3 text-center">
                    <h3 class="text-lg font-medium text-gray-900 mb-4">Dirección TRON (TRC20)</h3>
                    <div class="flex items-center justify-center space-x-2 mb-4">
                        <input type="text" value="${tronAddress}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="tronAddressInput">
                        <button onclick="copyTronAddress()" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
                            Copiar
                        </button>
                    </div>
                    <p class="text-lg text-red-600 font-bold mb-6">ADVERTENCIA: Si no realiza la inversión en los próximos 3 días, su cuenta será desactivada y tendrá que registrarse nuevamente.</p>
                    <button onclick="window.location.href='/'" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
                        Cerrar
                    </button>
                </div>
            </div>
        `;
        document.body.appendChild(warningModal);
    };
    
    // Function to copy TRON address
    window.copyTronAddress = function() {
        const input = document.getElementById('tronAddressInput');
        input.select();
        document.execCommand('copy');
        alert('Dirección TRON copiada al portapapeles');
    };
    
    // Function to close investment modal
    window.closeInvestmentModal = function () {
        if (investmentModal) {
            investmentModal.classList.add('hidden');
        }
    };

    // Close investment modal when clicking outside
    window.addEventListener('click', (e) => {
        if (e.target === investmentModal) {
            investmentModal.classList.add('hidden');
        }
    });
});