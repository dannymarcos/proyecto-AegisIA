document.addEventListener('DOMContentLoaded', () => {
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    const mobileSettingsSubmenu = document.getElementById('mobile-settings-submenu');

    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('show');
    });

    // Close mobile menu when clicking outside
    document.addEventListener('click', (event) => {
        if (!mobileMenu.contains(event.target) && !mobileMenuButton.contains(event.target)) {
            mobileMenu.classList.add('hidden');
            if (mobileSettingsSubmenu) {
                mobileSettingsSubmenu.classList.add('hidden');
            }
        }
    });

    // Handle language selection
    const languageSelect = document.getElementById('language-select');
    const mobileLanguageSelect = document.getElementById('mobile-language-select');

    async function handleLanguageChange(event) {
        const selectedLanguage = event.target.value;
        try {
            // First request to get confirmation
            const confirmResponse = await fetch('/change_language', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ language: selectedLanguage })
            });
            
            const confirmData = await confirmResponse.json();
            if (confirmData.status === 'confirm') {
                if (confirm(confirmData.message)) {
                    // Second request to actually change the language
                    const response = await fetch('/change_language', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ 
                            language: selectedLanguage,
                            confirm: true
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error('Failed to change language');
                    }
                    
                    const data = await response.json();
                    if (data.translations) {
                        // Update all text elements with translations
                        Object.keys(data.translations).forEach(key => {
                            const elements = document.querySelectorAll(`[data-translate="${key}"]`);
                            elements.forEach(element => {
                                element.textContent = data.translations[key];
                            });
                        });
                    }
                    
                    // Reload page to apply all translations
                    window.location.reload();
                }
            }
        } catch (error) {
            console.error('Error changing language:', error);
            alert('Failed to change language. Please try again.');
        }
    }

    if (languageSelect) {
        languageSelect.addEventListener('change', handleLanguageChange);
    }
    if (mobileLanguageSelect) {
        mobileLanguageSelect.addEventListener('change', handleLanguageChange);
    }
});

function toggleMobileSettings() {
    const submenu = document.getElementById('mobile-settings-submenu');
    submenu?.classList.toggle('hidden');
}