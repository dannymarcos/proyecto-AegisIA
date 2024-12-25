document.addEventListener('DOMContentLoaded', () => {
    const passwordForm = document.getElementById('password-form');
    const messageDiv = document.getElementById('message');
    
    function showMessage(message, isError = false) {
        messageDiv.textContent = message;
        messageDiv.className = isError ? 'text-red-600' : 'text-green-600';
        messageDiv.classList.remove('hidden');
        setTimeout(() => {
            messageDiv.classList.add('hidden');
        }, 5000);
    }
    
    passwordForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const currentPassword = document.getElementById('current_password').value;
        const newPassword = document.getElementById('new_password').value;
        const confirmPassword = document.getElementById('confirm_password').value;
        
        if (newPassword !== confirmPassword) {
            showMessage('Las contraseñas nuevas no coinciden', true);
            return;
        }
        
        try {
            const response = await fetch('/settings/password', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    current_password: currentPassword,
                    new_password: newPassword
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Contraseña actualizada exitosamente');
                passwordForm.reset();
            } else {
                showMessage(data.message || 'Error al actualizar la contraseña', true);
            }
        } catch (error) {
            console.error('Error updating password:', error);
            showMessage('Ocurrió un error al actualizar la contraseña', true);
        }
    });
});