document.addEventListener('keydown', function(e) {
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'TEXTAREA') {
        e.preventDefault();
        const input = document.getElementById('input-busca');
        if (input) input.focus();
    }
});

function setupAutocomplete(inputId, hiddenId, campo) {
    const input = document.getElementById(inputId);
    const hidden = hiddenId ? document.getElementById(hiddenId) : null;
    const list = document.getElementById(inputId + '-list');
    if (!input || !list) return;

    let timeout = null;

    input.addEventListener('input', function() {
        clearTimeout(timeout);
        const q = this.value;
        if (q.length < 2) {
            list.innerHTML = '';
            if (hidden) hidden.value = '';
            return;
        }
        timeout = setTimeout(async () => {
            const res = await fetch(`/api/sugerir?campo=${campo}&q=${encodeURIComponent(q)}`);
            const data = await res.json();
            list.innerHTML = '';
            data.forEach(item => {
                const div = document.createElement('div');
                div.textContent = item.texto;
                div.className = 'autocomplete-item';
                div.onclick = () => {
                    input.value = item.texto;
                    if (hidden) hidden.value = item.id;
                    list.innerHTML = '';
                    document.getElementById('form-busca').dispatchEvent(new Event('submit', {cancelable: true}));
                };
                list.appendChild(div);
            });
        }, 300);
    });

    document.addEventListener('click', function(e) {
        if (e.target !== input && e.target !== list) {
            list.innerHTML = '';
        }
    });
}
