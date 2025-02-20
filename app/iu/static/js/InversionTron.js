



export default class InversionTron {
    constructor() {
        this.tronAddress = "TRWzwYKqKvNgZxGgKEyYAqFvWwKFEQkXrk";  
        this.copyFixedTronAddress = this.copyFixedTronAddress.bind(this);
        this.showInvestmentWarning = this.showInvestmentWarning.bind(this);
    }      
    
    copyFixedTronAddress() {
        navigator.clipboard.writeText(this.tronAddress)
            .then(() => alert('Dirección TRON copiada al portapapeles'))
            .catch(() => alert('Error al copiar la dirección'));
    }
        
    showInvestmentWarning(investment_btn) {
        const d = document;
        const $investment_boton = d.getElementById(investment_btn);
        const $warningModal = document.createElement('div');
        
        $investment_boton.addEventListener("click", () => {
            $warningModal.className = 'altura fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full';
            $warningModal.innerHTML = `
                <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                    <div class="mt-3 text-center">
                        <h3 class="text-lg font-medium text-gray-900 mb-4">Dirección TRON (TRC20)</h3>
                        <div class="flex items-center justify-center space-x-2 mb-4">
                            <input type="text" value="${this.tronAddress}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="tronAddressInput">
                            <button id="copy-button" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
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
            d.body.appendChild($warningModal);

            // Añadir el event listener para el botón de copiar
            d.getElementById('copy-button').addEventListener('click', this.copyFixedTronAddress);
        });
    }
}
