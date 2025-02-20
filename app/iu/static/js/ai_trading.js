// Initialize trading variables
let isAITrading = false;
let tradingMode = 'spot';
let useCustomStrategies = false;
let customStrategy = '';

// Function to update AI trading info
function updateAITradingInfo(data) {
    // Update basic trading info
    const strategyElement = document.getElementById('strategy');
    
    // Update market analysis info
    if (data.market_analysis) {
        document.getElementById('current-price').textContent = data.market_analysis.current_price || '-';
        document.getElementById('price-change').textContent = data.market_analysis.price_change || '-';
        document.getElementById('rsi-value').textContent = data.market_analysis.rsi || '-';
        document.getElementById('macd-value').textContent = data.market_analysis.macd || '-';
        document.getElementById('stochastic-value').textContent = data.market_analysis.stochastic || '-';
        document.getElementById('stop-loss-value').textContent = data.market_analysis.stop_loss || '-';
        document.getElementById('take-profit-value').textContent = data.market_analysis.take_profit || '-';
        document.getElementById('patterns-value').textContent = data.market_analysis.patterns || '-';
        document.getElementById('signals-value').textContent = data.market_analysis.signals || '-';
    }
    
    if (strategyElement) strategyElement.textContent = data.strategy || 'No active strategy';
    
    const startPriceElement = document.getElementById('start-price');
    const endPriceElement = document.getElementById('end-price');
    const profitLossElement = document.getElementById('profit-loss');
    const actionElement = document.getElementById('ai-action');
    const statusElement = document.getElementById('ai-status');

    if (startPriceElement && data.start_price) startPriceElement.textContent = `$${parseFloat(data.start_price).toFixed(2)}`;
    if (endPriceElement && data.end_price) endPriceElement.textContent = `$${parseFloat(data.end_price).toFixed(2)}`;
    if (profitLossElement && data.performance) {
        profitLossElement.textContent = data.performance;
        const performanceValue = parseFloat(data.performance);
        profitLossElement.style.color = !isNaN(performanceValue) && performanceValue >= 0 ? '#008000' : '#FF0000';
    }
    if (actionElement) {
        actionElement.textContent = data.action ? data.action.toUpperCase() : '-';
        actionElement.style.color = data.action === 'buy' ? '#008000' : '#FF0000';
    }
    if (statusElement && data.status) {
        statusElement.textContent = data.error || data.message || '-';
        statusElement.style.color = data.status === 'success' ? '#008000' : 
                                   data.status === 'warning' ? '#FFA500' : '#FF0000';
    }
}

// Function to get trade settings
function getTradeSettings() {
    const tradeSizeType = document.getElementById('trade-size-type');
    const tradeSizeValue = document.getElementById('trade-size-value');
    const takeProfit = document.getElementById('take-profit');
    const stopLoss = document.getElementById('stop-loss');
    
    return {
        size_type: tradeSizeType ? tradeSizeType.value : 'fixed',
        size_value: tradeSizeValue ? parseFloat(tradeSizeValue.value) || 0 : 0,
        take_profit: takeProfit ? parseFloat(takeProfit.value) || 0 : 0,
        stop_loss: stopLoss ? parseFloat(stopLoss.value) || 0 : 0,
        trading_mode: tradingMode,
        use_custom_strategies: useCustomStrategies,
        custom_strategy: customStrategy
    };
}

// Function to start AI trading
async function startAITrading() {
    try {
        const settings = getTradeSettings();
        console.log('Starting AI trading with settings:', settings);

        const response = await fetch('/analyze_and_execute_strategy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings),
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error('Server response:', errorText);
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const data = await response.json();
        if (data.error) {
            console.error('Error from server:', data.error);
            alert(data.error);
            throw new Error(data.error);
        }

        updateAITradingInfo({
            strategy: data.strategy || 'No active strategy',
            start_price: data.initial_price || '-',
            end_price: data.current_price || '-',
            performance: data.performance || 0,
            action: data.action || '-',
            status: data.status || 'warning',
            message: data.message || 'Operation completed',
            market_analysis: {
                current_price: data.current_price || '-',
                price_change: data.price_change || '-',
                rsi: data.rsi || '-',
                macd: data.macd || '-',
                stochastic: `${data.stochastic_k?.toFixed(2)}/${data.stochastic_d?.toFixed(2)}` || '-',
                stop_loss: data.stop_loss || '-',
                take_profit: data.take_profit || '-',
                patterns: data.patterns || '-',
                signals: data.signals || '-'
            }
        });
        
        const aiTradingInfo = document.getElementById('ai-trading-info');
        if (aiTradingInfo) aiTradingInfo.classList.remove('hidden');
        console.log('AI Trading started successfully');
        
        // Start monitoring the market every 15 minutes
        setInterval(() => {
            fetch('/analyze_and_execute_strategy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(getTradeSettings()),
            }).then(response => response.json())
              .then(data => {
                  if (data.error) {
                      console.error('Error from server:', data.error);
                  } else {
                      updateAITradingInfo({
                          strategy: data.strategy || 'No active strategy',
                          start_price: data.start_price || '-',
                          end_price: data.end_price || '-',
                          performance: data.performance || 0,
                          action: data.action || '-',
                          status: data.status || 'warning',
                          message: data.message || 'Operation completed',
                          market_analysis: {
                              current_price: data.current_price || '-',
                              price_change: data.price_change || '-',
                              rsi: data.rsi || '-',
                              macd: data.macd || '-',
                              stochastic: `${data.stochastic_k?.toFixed(2)}/${data.stochastic_d?.toFixed(2)}` || '-',
                              stop_loss: data.stop_loss || '-',
                              take_profit: data.take_profit || '-',
                              patterns: data.patterns || '-',
                              signals: data.signals || '-'
                          }
                      });
                  }
              })
              .catch(error => console.error('Error during market monitoring:', error));
        }, 900000); // 15 minutes in milliseconds
    } catch (error) {
        console.error('Error during AI trading execution:', error);
        const errorMessage = error.message.includes('HTTP error') ? 
            'Error starting trading. Please verify your balance and settings.' :
            `Error starting trading: ${error.message}`;
        alert(errorMessage);
    }
}

// Function to stop AI trading
function stopAITrading() {
    fetch('/stop_ai_trading', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    })
        .then((response) => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then((data) => {
            if (data.status === 'success') {
                updateAITradingInfo({
                    strategy: '-',
                    start_price: '-',
                    end_price: '-',
                    profit_loss: 0,
                    market_analysis: {
                        current_price: '-',
                        price_change: '-',
                        rsi: '-',
                        macd: '-',
                        stochastic: '-',
                        stop_loss: '-',
                        take_profit: '-',
                        patterns: '-',
                        signals: '-'
                    }
                });
                const aiTradingInfo = document.getElementById('ai-trading-info');
                if (aiTradingInfo) aiTradingInfo.classList.add('hidden');
                console.log('AI Trading stopped successfully');
            } else {
                throw new Error('Failed to stop AI Trading');
            }
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('Error stopping trading. Please try again.');
        });
}

// Event listeners for buttons
document.addEventListener('DOMContentLoaded', () => {
    const aiTradingButton = document.getElementById('ai-trading-button');
    if (aiTradingButton) {
        aiTradingButton.addEventListener('click', () => {
            if (isAITrading) {
                stopAITrading();
                aiTradingButton.textContent = 'Authorize AI Trading';
                aiTradingButton.classList.remove('bg-red-500', 'hover:bg-red-700');
                aiTradingButton.classList.add('bg-green-500', 'hover:bg-green-700');
            } else {
                startAITrading();
                aiTradingButton.textContent = 'Stop AI Trading';
                aiTradingButton.classList.remove('bg-green-500', 'hover:bg-green-700');
                aiTradingButton.classList.add('bg-red-500', 'hover:bg-red-700');
            }
            isAITrading = !isAITrading;
        });
    }
});

// Function to update trading mode
function updateTradingMode(mode) {
    tradingMode = mode;
    console.log(`Trading mode updated to: ${tradingMode}`);
}

// Event listener for save strategy button
const sendStrategyButton = document.getElementById('send-strategy');
if (sendStrategyButton) {
    sendStrategyButton.addEventListener('click', () => {
        const strategyInput = document.getElementById('chat-input').value;
        if (strategyInput.trim().toLowerCase().endsWith('guardar estrategias')) {
            customStrategy = strategyInput;
            useCustomStrategies = true;
            
            // Update strategy mode button
            const strategyModeButton = document.getElementById('strategy-mode-button');
            if (strategyModeButton) {
                strategyModeButton.textContent = 'Using Custom Strategy';
                strategyModeButton.classList.remove('bg-green-500', 'hover:bg-green-700');
                strategyModeButton.classList.add('bg-blue-500', 'hover:bg-blue-700');
            }
            
            // Add message to chat
            const chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'text-right';
                messageDiv.textContent = 'Strategy saved successfully!';
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
            
            // Clear input
            document.getElementById('chat-input').value = '';
            
            console.log('Strategy saved:', customStrategy);
            console.log('Using custom strategies:', useCustomStrategies);
        }
    });
}

// Event listener for strategy mode button
const strategyModeButton = document.getElementById('strategy-mode-button');
if (strategyModeButton) {
    strategyModeButton.addEventListener('click', () => {
        useCustomStrategies = !useCustomStrategies;
        const chatMessages = document.getElementById('chat-messages');
        if (chatMessages) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'text-right';
            
            if (useCustomStrategies && customStrategy) {
                strategyModeButton.textContent = 'Using Custom Strategy';
                strategyModeButton.classList.remove('bg-green-500', 'hover:bg-green-700');
                strategyModeButton.classList.add('bg-blue-500', 'hover:bg-blue-700');
                messageDiv.textContent = 'Switched to using custom strategy';
            } else {
                strategyModeButton.textContent = 'Using All Strategies';
                strategyModeButton.classList.remove('bg-blue-500', 'hover:bg-blue-700');
                strategyModeButton.classList.add('bg-green-500', 'hover:bg-green-700');
                messageDiv.textContent = 'Switched to using all strategies';
            }
            
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
            console.log(`Strategy mode updated: ${useCustomStrategies ? 'Custom' : 'All'}`);
        }
    });
}

// Event listener for send strategy button
if (document.getElementById('send-strategy')) {
    document.getElementById('send-strategy').addEventListener('click', () => {
        const chatInput = document.getElementById('chat-input');
        const chatMessages = document.getElementById('chat-messages');
        
        if (chatInput && chatMessages) {
            const strategyInput = chatInput.value;
            if (strategyInput.trim().toLowerCase().endsWith('guardar estrategias')) {
                customStrategy = strategyInput;
                useCustomStrategies = true;
                
                // Update strategy mode button
                if (strategyModeButton) {
                    strategyModeButton.textContent = 'Using Custom Strategy';
                    strategyModeButton.classList.remove('bg-green-500', 'hover:bg-green-700');
                    strategyModeButton.classList.add('bg-blue-500', 'hover:bg-blue-700');
                }
                
                // Add message to chat
                const messageDiv = document.createElement('div');
                messageDiv.className = 'text-right';
                messageDiv.textContent = 'Strategy saved successfully!';
                chatMessages.appendChild(messageDiv);
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Clear input
                chatInput.value = '';
                
                console.log('Strategy saved:', customStrategy);
                console.log('Using custom strategies:', useCustomStrategies);
            }
        }
    });
}
// Keeping the implementation in the first declaration