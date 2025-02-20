
import TvWidget from "./js/TvWidget.js";
import get_cryptos from "./js/get_cryptos.js";
import InversionTron from "./js/InversionTron.js";
import ReferralLink from "./js/ReferralLink.js";
import method_trading from "./js/MethodTrading.js";

import get_method from "./js/get_method.js";

const d = document;
const ls = localStorage;

d.addEventListener("DOMContentLoaded",async (e)=>{
    const tvWidget = new TvWidget();
    const inversion = new InversionTron();
    const link_ref = new ReferralLink();
    
    let method = ls.getItem('method');
    // Si el localStorage esta vacio, obtiene el método por defecto desde el backend y lo almacena
    if(!method){
        method = await get_method();
        if(method){
            ls.setItem('method',method);
        }else{
         //Si no se pudo obtener el método por algun error. usar "spot" como callback
         method = 'spot';
         ls.setItem('method',method)
        }
    }
    //Clase que permite cargar el widget de trading view en el cuerpo del documento
    tvWidget.loadTradingSymbol().then(() => {
        // console.log(`El símbolo actual es: ${tvWidget.getCurrentTradingSymbol()}`); 
        
        // console.log(`El símbolo actual guardado en localStorage es: ${localStorage.getItem("symbol")}`);
    });
    
    //Inversion en la cuenta de tron
    inversion.showInvestmentWarning('investment-btn');
    //Generar link de referidos
    link_ref.detectarClick('generate-referral-btn')
    
    //Funcion que permite obtener una lista detallada de los pares de divisas disponibles en la API de kraken, dependiendo del metodo
    // de trading que este almacenado en la variable method
    get_cryptos(method,'fetchCryptos','crypto-list','crypto-search');
   
    //Esta funcion permite mediante el boton 'trading-mode-button' cambiar entre los controles
    // para operar en spot y en futuros, dependiendo del metodo en el que se encuentre
    method_trading('trading-mode-button','manual-trading-controls','ai-trading-controls');

});
