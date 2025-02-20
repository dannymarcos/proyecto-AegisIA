document.addEventListener('DOMContentLoaded', () => {
    const withdrawalForm = document.getElementById('withdrawal-form');
    const currencySelect = document.getElementById('withdrawal-currency');
    const walletTypeSpan = document.getElementById('wallet-type');
    
    if (currencySelect) {
        currencySelect.addEventListener('change', () => {
            const selectedCurrency = currencySelect.value;
            walletTypeSpan.textContent = selectedCurrency === 'USDT' ? 'USDT (TRC20)' : 'BTC';
        });
    }
    
    if (withdrawalForm) {
        withdrawalForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const amount = document.getElementById('withdrawal-amount').value;
            const currency = document.getElementById('withdrawal-currency').value;
            const walletAddress = document.getElementById('wallet-address').value;
            
            try {
                const response = await fetch('/request_withdrawal', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ 
                        amount, 
                        currency,
                        wallet_address: walletAddress 
                    })
                });
                
                const data = await response.json();
                if (data.status === 'success' && data.redirect) {
                    window.location.href = data.redirect;
                } else {
                    alert(data.message || 'Failed to submit withdrawal request');
                }
            } catch (error) {
                console.error('Error submitting withdrawal request:', error);
                alert('An error occurred while submitting your withdrawal request');
            }
        });
    }
});