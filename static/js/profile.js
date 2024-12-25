document.addEventListener('DOMContentLoaded', () => {
    const profileForm = document.getElementById('profile-form');
    const messageDiv = document.getElementById('message');
    
    function showMessage(message, isError = false) {
        messageDiv.textContent = message;
        messageDiv.className = isError ? 'error-message' : 'success-message';
        messageDiv.style.display = 'block';
        setTimeout(() => {
            messageDiv.style.display = 'none';
        }, 5000);
    }
    
    profileForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(profileForm);
        
        try {
            const response = await fetch('/profile', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                showMessage('Profile updated successfully!');
                // Clear password fields
                document.getElementById('current_password').value = '';
                document.getElementById('new_password').value = '';
            } else {
                showMessage(data.message || 'Failed to update profile', true);
            }
        } catch (error) {
            console.error('Error updating profile:', error);
            showMessage('An error occurred while updating your profile', true);
        }
    });
    
    // Add event listener for email change
    const emailInput = document.getElementById('email');
    const originalEmail = emailInput.value;
    
    emailInput.addEventListener('change', () => {
        if (emailInput.value !== originalEmail) {
            const confirmChange = confirm('Changing your email will require verification. Continue?');
            if (!confirmChange) {
                emailInput.value = originalEmail;
            }
        }
    });
});