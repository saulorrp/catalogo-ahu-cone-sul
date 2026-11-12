// catalogo.js — o pouco de JavaScript da interface. Tudo por delegação no documento: os
// filtros e a gaveta da seleção são trocados pelo HTMX, e o que é trocado não perde o comportamento.

(function () {
    'use strict';

    const form = () => document.getElementById('form-busca');
    const enviar = () => { const f = form(); if (f && window.htmx) htmx.trigger(f, 'submit'); else if (f) f.submit(); };

    // "/" leva o cursor à busca.
    document.addEventListener('keydown', function (e) {
        const el = document.activeElement;
        const digitando = el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT');
        if (e.key === '/' && !digitando) {
            const input = document.getElementById('input-busca');
            if (input) { e.preventDefault(); input.focus(); input.select(); }
        }
    });

    // Os campos vazios do formulário não vão para o endereço (a1=&a2=&fu=…).
    document.addEventListener('htmx:configRequest', function (e) {
        const p = e.detail.parameters;
        if (!p || typeof p.forEach !== 'function') return;
        const vazios = [];
        p.forEach(function (valor, chave) { if (valor === '') vazios.push(chave); });
        vazios.forEach(function (chave) { p.delete(chave); });
    });

    document.addEventListener('click', function (e) {
        // Barra de década no filtro de período: preenche o intervalo e busca.
        const barra = e.target.closest('[data-decada]');
        if (barra) {
            const d = parseInt(barra.dataset.decada, 10);
            const de = document.getElementById('f-a1'), ate = document.getElementById('f-a2');
            if (de && ate) { de.value = d; ate.value = d + 9; enviar(); }
            return;
        }
        // Abrir e fechar os filtros na tela estreita.
        const alterna = e.target.closest('[data-alterna]');
        if (alterna) {
            const alvo = document.querySelector(alterna.dataset.alterna);
            if (alvo) alvo.classList.toggle('aberto');
            return;
        }
        // Copiar a cota, com a confirmação no próprio botão.
        const copiar = e.target.closest('[data-copiar]');
        if (copiar && navigator.clipboard) {
            navigator.clipboard.writeText(copiar.dataset.copiar).then(function () {
                const antes = copiar.textContent;
                copiar.textContent = '✓ Cota copiada';
                setTimeout(function () { copiar.textContent = antes; }, 1800);
            });
            return;
        }
        // Escolher uma sugestão do autocompletar.
        const item = e.target.closest('.autocomplete-item');
        if (item) {
            const caixa = item.closest('.autocomplete-container');
            const input = caixa.querySelector('[data-sugerir]');
            const oculto = input.dataset.oculto ? document.getElementById(input.dataset.oculto) : null;
            input.value = item.textContent;
            if (oculto) oculto.value = item.dataset.id;
            fecharLista(caixa);
            enviar();
            return;
        }
        document.querySelectorAll('.autocomplete-container').forEach(function (caixa) {
            if (!caixa.contains(e.target)) fecharLista(caixa);
        });
    });

    // Autocompletar de remetente e destinatário.
    function fecharLista(caixa) {
        const lista = caixa.querySelector('.autocomplete-list');
        if (lista) { lista.innerHTML = ''; lista.hidden = true; }
    }

    let espera = null;
    document.addEventListener('input', function (e) {
        const input = e.target.closest('[data-sugerir]');
        if (!input) return;
        const caixa = input.closest('.autocomplete-container');
        const lista = caixa.querySelector('.autocomplete-list');
        const oculto = input.dataset.oculto ? document.getElementById(input.dataset.oculto) : null;
        clearTimeout(espera);
        const q = input.value.trim();
        if (q.length < 2) {
            fecharLista(caixa);
            // Campo apagado: o filtro sai.
            if (oculto && oculto.value) { oculto.value = ''; enviar(); }
            return;
        }
        espera = setTimeout(async function () {
            const base = document.body.dataset.urlSugerir;
            const res = await fetch(base + '?campo=' + input.dataset.sugerir + '&q=' + encodeURIComponent(q));
            if (!res.ok) return;
            const dados = await res.json();
            lista.innerHTML = '';
            dados.forEach(function (d) {
                const div = document.createElement('div');
                div.className = 'autocomplete-item';
                div.textContent = d.texto;
                div.dataset.id = d.id;
                lista.appendChild(div);
            });
            lista.hidden = dados.length === 0;
        }, 250);
    });

    // A gaveta da seleção lembra, nesta aba do navegador, se a pessoa a fechou ou abriu.
    const CHAVE = 'ahu-gaveta-selecao';
    function lembrar() { try { return sessionStorage.getItem(CHAVE); } catch (_) { return null; } }
    function aplicarGaveta() {
        const g = document.getElementById('gaveta-selecao');
        if (!g) return;
        const escolha = lembrar();
        if (escolha === 'aberta') g.open = true;
        else if (escolha === 'fechada') g.open = false;
        else if (window.matchMedia('(max-width: 1180px)').matches) g.open = false;   // estreita: começa fechada
    }
    document.addEventListener('toggle', function (e) {
        if (e.target.id !== 'gaveta-selecao' || e.target.dataset.silencio) return;
        try { sessionStorage.setItem(CHAVE, e.target.open ? 'aberta' : 'fechada'); } catch (_) { /* sem armazenamento */ }
    }, true);
    function aplicarEmSilencio() {
        const g = document.getElementById('gaveta-selecao');
        if (!g) return;
        g.dataset.silencio = '1';
        aplicarGaveta();
        setTimeout(function () { delete g.dataset.silencio; }, 150);
    }
    document.addEventListener('DOMContentLoaded', aplicarEmSilencio);
    document.addEventListener('htmx:oobAfterSwap', function (e) {
        if (e.detail.target && e.detail.target.id === 'gaveta-selecao') aplicarEmSilencio();
    });
})();
