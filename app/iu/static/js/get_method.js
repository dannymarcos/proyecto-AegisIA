// app/iu/static/js/get_method.js
export default async function get_method() {

    try {
        let response = await fetch('/get_method_trading');
        let json = await response.json();

        console.log(`Respuesta desde /get_method_trading en routes.py:`, json);

        // Verificar si hay errores en la respuesta
        if (json.error && json.error.length > 0) {
            console.error('Error en la respuesta de la API de Kraken:', json.error);
            return;
        }

        return json.get_method;
    }catch(error){
        console.error('Error:', error);
        return null;
    }
    
}