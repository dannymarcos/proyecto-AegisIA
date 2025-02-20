export default class ReferralLink {
    constructor() {
        // Asegurarse de que el contexto `this` esté enlazado correctamente
        this.detectarClick = this.detectarClick.bind(this);
        this.copyReferralLink = this.copyReferralLink.bind(this);
    }

    async detectarClick(boton) {
        const d = document;
        const $button_ref = d.getElementById(boton);

        if ($button_ref) {
            $button_ref.addEventListener("click", async (e) => {
                console.log("Se detectó el click dentro del método ReferralLink.detectarClick", e.target);

                try {
                    const response = await fetch('/generate_referral_link');
                    const data = await response.json();
                    console.log("Estamos dentro de la clase ReferralLink y estamos obteniendo el archivo json que se genera en el enlace", data);
                    
                    if (data.status === 'success') {
                        this.referral_link = data.referral_link;  // Almacenar el enlace de referencia

                        const modalContent = `
                            <div class="altura fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full">
                                <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                                    <div class="mt-3 text-center">
                                        <h3 class="text-lg font-medium text-gray-900 mb-4">Tu Link de Referido</h3>
                                        <div class="flex items-center justify-center space-x-2 mb-4">
                                            <input type="text" value="${data.referral_link}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="referralLinkInput">
                                            <button id="copy-button" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
                                                Copiar
                                            </button>
                                        </div>
                                        <p class="text-sm text-gray-600 mb-4">Total de referidos: ${data.total_referrals}</p>
                                        <button id="closeReferralModal" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
                                            Cerrar
                                        </button>
                                    </div>
                                </div>
                            </div>`;
                        
                        // Añadir el modal al cuerpo del documento
                        const modalDiv = document.createElement('div');
                        modalDiv.id = 'referralLinkModal';
                        modalDiv.innerHTML = modalContent;
                        d.body.appendChild(modalDiv);

                        // Añadir la función de copia al botón
                        const copyButton = d.getElementById('copy-button');
                        copyButton.addEventListener('click', this.copyReferralLink);

                        //Añadir la funcion de cerrar al boton
                        const closeButton = d.getElementById('closeReferralModal');
                        closeButton.addEventListener('click',this.closeReferralModal);
            
                    } else {
                        alert('Error generando link de referido: ' + (data.message || 'Error desconocido'));
                    }
                } catch (error) {
                    console.error('Error:', error);
                    alert('Error generando link de referido. Por favor intenta nuevamente.');
                }
            });
        } else {
            console.error(`No se encontró el botón con id: ${boton}`);
        }
    }

    copyReferralLink() {
        const input = document.getElementById('referralLinkInput');
        input.select();
        document.execCommand('copy');
        alert('Link copiado al portapapeles!');
    }

    //   Add close function
    closeReferralModal = function() {
        const modal = document.getElementById('referralLinkModal');
            if (modal) {
                 modal.remove();
                }
            };
}



    // showReferralModal = async function() {
    //     try {
    //         const response = await fetch('/generate_referral_link');
    //         const data = await response.json();
            
    //         if (data.status === 'success') {
    //             const modalContent = `
    //                 <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full">
    //                     <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
    //                         <div class="mt-3 text-center">
    //                             <h3 class="text-lg font-medium text-gray-900 mb-4">Tu Link de Referido</h3>
    //                             <div class="flex flex-col space-y-4">
    //                                 <div class="flex items-center justify-center space-x-2">
    //                                     <input type="text" value="${data.referral_link}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="referralLinkInput">
    //                                     <button onclick="copyReferralLink()" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
    //                                         Copiar
    //                                     </button>
    //                                 </div>
    //                                 <div class="referral-tree">
    //                                     <h4 class="text-md font-medium text-gray-800 mb-2">Árbol de Referidos</h4>
    //                                     <div id="referralTreeContainer" class="max-h-60 overflow-y-auto p-2 border rounded">
    //                                         <ul class="list-none">
    //                                             <li class="mb-2">
    //                                                 <span class="font-medium">Tus Referidos Directos:</span>
    //                                                 <ul class="pl-4 mt-1">
    //                                                     ${data.direct_referrals ? data.direct_referrals.map(ref => 
    //                                                         `<li class="text-sm text-gray-600">${ref.name} - ${ref.date}</li>`
    //                                                     ).join('') : '<li class="text-sm text-gray-500">No hay referidos directos aún</li>'}
    //                                                 </ul>
    //                                             </li>
    //                                             <li>
    //                                                 <span class="font-medium">Referidos de tus Referidos:</span>
    //                                                 <ul class="pl-4 mt-1">
    //                                                     ${data.indirect_referrals ? data.indirect_referrals.map(ref => 
    //                                                         `<li class="text-sm text-gray-600">${ref.name} - ${ref.date} (Referido por: ${ref.referred_by})</li>`
    //                                                     ).join('') : '<li class="text-sm text-gray-500">No hay referidos indirectos aún</li>'}
    //                                                 </ul>
    //                                             </li>
    //                                         </ul>
    //                                     </div>
    //                                 </div>
    //                                 <p class="text-sm text-gray-600">Total de referidos: ${data.total_referrals}</p>
    //                                 <button onclick="closeReferralModal()" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
    //                                     Cerrar
    //                                 </button>
    //                             </div>
    //                         </div>
    //                     </div>
    //                 </div>`;
                
    //             // Add modal to body
    //             const modalDiv = document.createElement('div');
    //             modalDiv.id = 'referralLinkModal';
    //             modalDiv.innerHTML = modalContent;
    //             document.body.appendChild(modalDiv);
                
    //             // Add copy function
    //             window.copyReferralLink = function() {
    //                 const input = document.getElementById('referralLinkInput');
    //                 input.select();
    //                 document.execCommand('copy');
    //                 alert('Link copiado al portapapeles!');
    //             };
                
    //             // Add close function
    //             window.closeReferralModal = function() {
    //                 const modal = document.getElementById('referralLinkModal');
    //                 if (modal) {
    //                     modal.remove();
    //                 }
    //             };
    //         } else {
    //             alert('Error generando link de referido: ' + (data.message || 'Error desconocido'));
    //         }
    //     } catch (error) {
    //         console.error('Error:', error);
    //         alert('Error generando link de referido. Por favor intenta nuevamente.');
    //     }
    //     try {
    //         const response = await fetch('/generate_referral_link');
    //         const data = await response.json();
            
    //         if (data.status === 'success') {
    //             const modalContent = `
    //                 <div class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full">
    //                     <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
    //                         <div class="mt-3 text-center">
    //                             <h3 class="text-lg font-medium text-gray-900 mb-4">Tu Link de Referido</h3>
    //                             <div class="flex items-center justify-center space-x-2 mb-4">
    //                                 <input type="text" value="${data.referral_link}" readonly class="w-full p-2 border rounded bg-gray-50 text-sm" id="referralLinkInput">
    //                                 <button onclick="copyReferralLink()" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-300">
    //                                     Copiar
    //                                 </button>
    //                             </div>
    //                             <p class="text-sm text-gray-600 mb-4">Total de referidos: ${data.total_referrals}</p>
    //                             <button onclick="closeReferralModal()" class="w-full px-4 py-2 bg-red-500 text-white text-base font-medium rounded-md shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-300">
    //                                 Cerrar
    //                             </button>
    //                         </div>
    //                     </div>
    //                 </div>`;
                
    //             // Add modal to body
    //             const modalDiv = document.createElement('div');
    //             modalDiv.id = 'referralLinkModal';
    //             modalDiv.innerHTML = modalContent;
    //             document.body.appendChild(modalDiv);
                
    //             // Add copy function
    //             window.copyReferralLink = function() {
    //                 const input = document.getElementById('referralLinkInput');
    //                 input.select();
    //                 document.execCommand('copy');
    //                 alert('Link copiado al portapapeles!');
    //             };
                
    //             // Add close function
    //             window.closeReferralModal = function() {
    //                 const modal = document.getElementById('referralLinkModal');
    //                 if (modal) {
    //                     modal.remove();
    //                 }
    //             };
    //         } else {
    //             alert('Error generando link de referido: ' + (data.message || 'Error desconocido'));
    //         }
    //     } catch (error) {
    //         console.error('Error:', error);
    //         alert('Error generando link de referido. Por favor intenta nuevamente.');
    //     }
    //     toggleReferralMenu();
    // };
    



// }