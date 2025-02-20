export default function method_trading(boton, manualTradingButton, aiTradingButton) {
    const d = document;
    const $spotButton = d.getElementById(boton);
    const $manTradBut = d.getElementById(manualTradingButton);
    const $aiTradBut = d.getElementById(aiTradingButton);
    const ls = localStorage;

    // Verifica si hay un método de trading almacenado en el localStorage
    let storedMethod = ls.getItem('method');

    // Inicializar el estado de la página basado en el valor almacenado en localStorage
    if (storedMethod === 'Trade in Futures') {
        $aiTradBut.classList.remove('hidden');
        $manTradBut.classList.add('hidden');
        $spotButton.textContent = "Trade in Spot";
    } else {
        $manTradBut.classList.remove('hidden');
        $aiTradBut.classList.add('hidden');
        $spotButton.textContent = "Trade in Futures";
    }

    $spotButton.addEventListener("click", (e) => {
        console.log("Estamos dentro del botón ", e.target);

        // Determinar que valor está en el local storage y alternar entre los modos
        if ($spotButton.textContent === 'Trade in Futures') {
            // Cambiar a modo de trading en futuros
            $aiTradBut.classList.remove('hidden');
            $manTradBut.classList.add('hidden');
            $spotButton.textContent = "Trade in Spot";
            ls.setItem("method", "futures");
            console.log("Cambiamos a transacciones de futuros");
        } else {
            // Cambiar a modo de trading en spot
            $manTradBut.classList.remove('hidden');
            $aiTradBut.classList.add('hidden');
            $spotButton.textContent = "Trade in Futures";
            ls.setItem("method", "spot");
            console.log("Cambiamos a transacciones spot");
        }

        console.log($spotButton.textContent);
    });
}


