// Global alert listener — activo en todas las páginas
(function () {
  const evtSource = new EventSource('/events');

  evtSource.onmessage = (e) => {
    const { alert } = JSON.parse(e.data);
    const dot = document.querySelector('#navAlert .status-dot');
    const lbl = document.querySelector('#navAlert .status-label');
    if (!dot || !lbl) return;
    dot.className  = 'status-dot ' + (alert ? 'danger' : 'ok');
    lbl.textContent = alert ? 'ALERTA ACTIVA' : 'Sistema Activo';
  };

  evtSource.onerror = () => {
    const dot = document.querySelector('#navAlert .status-dot');
    if (dot) dot.className = 'status-dot';
  };
})();
