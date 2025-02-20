document.addEventListener('DOMContentLoaded', () => {
    const walletForm = document.getElementById('wallet-form');
    const messageDiv = document.getElementById('message');
    
    function showMessage(message, isError = false) {
        messageDiv.textContent = message;
        messageDiv.className = isError ? 'text-red-600' : 'text-green-600';
        messageDiv.classList.remove('hidden');
        setTimeout(() => {
            messageDiv.classList.add('hidden');
        }, 5000);
    }
    
    walletForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(walletForm);
        
        try {
            const response = await fetch('/settings/wallet', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Billetera actualizada exitosamente');
                document.getElementById('current_password').value = '';
            } else {
                showMessage(data.message || 'Error al actualizar la billetera', true);
            }
        } catch (error) {
            console.error('Error updating wallet:', error);
            showMessage('Ocurrió un error al actualizar la billetera', true);
        }
    });
});