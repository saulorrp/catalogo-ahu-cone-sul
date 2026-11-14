"""Varredura do DocReader do Projeto Resgate: página (pagfis) onde começa cada documento.

Cada página do DocReader traz o rótulo "Cx. N\\Doc. N"; o script acha, por busca galopante e
binária, a primeira página de cada documento. Precisa de IP brasileiro (VPN ligada).

Saída nesta pasta: resgate_pagfis_<bib>.tsv (bib, caixa, doc, pagfis) quando a coleção termina;
parcial_<bib>_w<n>.tsv enquanto roda. Pode ser interrompido (feche a janela) e retomado: rode de
novo e ele continua de onde parou.
"""
import os, re, sys, time, threading, urllib.request, urllib.error, ssl
from concurrent.futures import ThreadPoolExecutor

PASTA = os.path.dirname(os.path.abspath(__file__))
FILA = ['017_RJ_AV', '023_SP_AV', '012_CSRP', '003_BG_AV', '011_MG', '015_PE', '005_BA_AV',
        '005_BA_CA', '005_BA_LF', '009_MA', '013_PA', '050_CS_AV', '084_OU_AV', '030_SP_AV', '089_UL_AV']
TRABALHADORES = 8
URL = 'https://resgate.bn.gov.br/docreader/DocReader.aspx?bib={bib}&hf=resgate.bn.gov.br&pagfis={p}'
CAB = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                     'Chrome/142.0.0.0 Safari/537.36',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Language': 'pt-BR,pt;q=0.9'}
RE_PAG = re.compile(r'id="hPagFis"[^>]*value="(\d+)"')
RE_LAB = re.compile(r'Cx\.\s*(\d+)\\+Doc\.\s*([^<"]*)')
_trava = threading.Lock()
_cont = {'req': 0, 'docs': 0}


def log(msg):
    with _trava:
        print(time.strftime('%H:%M:%S'), msg, flush=True)


def rotulo(bib, p):
    """(ok, chave, caixa, doc) da página p. Espera e tenta de novo se a BN bloquear ou a rede cair."""
    espera = 5
    while True:
        try:
            req = urllib.request.Request(URL.format(bib=bib, p=p), headers=CAB)
            with urllib.request.urlopen(req, timeout=60) as r:
                h = r.read().decode('utf-8', 'ignore')
            _cont['req'] += 1
            mp, ml = RE_PAG.search(h), RE_LAB.search(h)
            ok = bool(mp and ml and int(mp.group(1)) == p)
            if not ml:
                return False, '', '', ''
            cx, doc = ml.group(1), ml.group(2).strip()
            return ok, cx + '|' + doc, cx, doc
        except urllib.error.HTTPError as e:
            if e.code == 403:
                log(f'403 da BN (VPN desligada?). Tento de novo em {espera}s.')
            else:
                log(f'HTTP {e.code}; tento de novo em {espera}s.')
        except Exception as e:
            log(f'erro de rede ({type(e).__name__}); tento de novo em {espera}s.')
        time.sleep(espera)
        espera = min(espera * 2, 120)


def max_pag(bib):
    arq = os.path.join(PASTA, f'max_{bib}.txt')
    if os.path.exists(arq):
        return int(open(arq).read())
    lo, hi = 1, 1024
    while rotulo(bib, hi)[0]:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        m = (lo + hi) // 2
        if rotulo(bib, m)[0]:
            lo = m
        else:
            hi = m
    open(arq, 'w').write(str(lo))
    return lo


def proximo(bib, P, chave, mx, passo):
    """Primeira página depois de P com rótulo diferente: (página, chave, caixa, doc) ou None no fim."""
    lo, s, hi, achado = P, passo, None, None
    while True:
        q = lo + s
        if q > mx:
            hi = mx + 1
            break
        r = rotulo(bib, q)
        if r[1] == chave:
            lo, s = q, s * 2
        else:
            hi, achado = q, r
            break
    while hi - lo > 1:
        m = (lo + hi) // 2
        r = rotulo(bib, m)
        if r[1] == chave:
            lo = m
        else:
            hi, achado = m, r
    if hi > mx or not achado or not achado[0]:
        return None
    return hi, achado[1], achado[2], achado[3]


def trabalhador(bib, i, a, b, mx):
    arq = os.path.join(PASTA, f'parcial_{bib}_w{i}.tsv')
    feito = os.path.join(PASTA, f'parcial_{bib}_w{i}.ok')
    if os.path.exists(feito):
        return
    linhas = []
    if os.path.exists(arq):  # ignora uma última linha cortada, se a janela fechou no meio da gravação
        linhas = [c for c in (l.rstrip('\n').split('\t') for l in open(arq, encoding='utf-8'))
                  if len(c) == 3 and c[0].isdigit() and c[2].isdigit()]
        with open(arq, 'w', encoding='utf-8') as g:
            g.write(''.join('\t'.join(c) + '\n' for c in linhas))
        if linhas:
            log(f'{bib} trecho {i}: retomando depois de Cx. {linhas[-1][0]} Doc. {linhas[-1][1]}')
    f = open(arq, 'a', encoding='utf-8')
    passo = 3
    if linhas:  # retoma do último documento gravado
        cx, doc, p = linhas[-1]
        P, chave = int(p), cx + '|' + doc
        r = proximo(bib, P, chave, mx, passo)
        if not r:
            open(feito, 'w').close(); return
        P, chave, cx, doc = r
    else:
        ok, chave, cx, doc = rotulo(bib, a)
        P = a
        if a > 1 and rotulo(bib, a - 1)[1] == chave:  # o documento começou no trecho anterior
            r = proximo(bib, a, chave, mx, passo)
            if not r:
                open(feito, 'w').close(); return
            P, chave, cx, doc = r
    while P <= b:
        f.write(f'{cx}\t{doc}\t{P}\n'); f.flush()
        _cont['docs'] += 1
        r = proximo(bib, P, chave, mx, passo)
        if not r:
            break
        passo = max(2, min(8, round((r[0] - P) * 0.8)))
        P, chave, cx, doc = r
    f.close()
    open(feito, 'w').close()


def colecao(bib):
    final = os.path.join(PASTA, f'resgate_pagfis_{bib}.tsv')
    if os.path.exists(final):
        log(f'{bib}: já feito, pulo.'); return
    mx = max_pag(bib)
    log(f'{bib}: {mx} páginas; começando.')
    tam = -(-mx // TRABALHADORES)
    with ThreadPoolExecutor(TRABALHADORES) as ex:
        fut = [ex.submit(trabalhador, bib, i, 1 + i * tam, min(mx, (i + 1) * tam), mx)
               for i in range(TRABALHADORES) if 1 + i * tam <= mx]
        while any(not x.done() for x in fut):
            time.sleep(60)
            log(f'{bib}: {_cont["docs"]} documentos, {_cont["req"]} páginas lidas nesta sessão')
        for x in fut:
            x.result()
    linhas = []
    for i in range(TRABALHADORES):
        arq = os.path.join(PASTA, f'parcial_{bib}_w{i}.tsv')
        if os.path.exists(arq):
            linhas += [l.rstrip('\n').split('\t') for l in open(arq, encoding='utf-8') if l.strip()]
    vistos, saida = set(), []
    for cx, doc, p in sorted(linhas, key=lambda x: int(x[2])):
        if p not in vistos:
            vistos.add(p); saida.append(f'{bib}\t{cx}\t{doc}\t{p}')
    with open(final, 'w', encoding='utf-8') as f:
        f.write('bib\tcaixa\tdoc\tpagfis\n' + '\n'.join(saida) + '\n')
    for i in range(TRABALHADORES):
        for ext in ('tsv', 'ok'):
            try: os.remove(os.path.join(PASTA, f'parcial_{bib}_w{i}.{ext}'))
            except FileNotFoundError: pass
    log(f'{bib}: PRONTO, {len(saida)} documentos -> {os.path.basename(final)}')


if __name__ == '__main__':
    try:
        req = urllib.request.Request(URL.format(bib='021_SC', p=37), headers=CAB)
        with urllib.request.urlopen(req, timeout=60) as r:
            h = r.read().decode('utf-8', 'ignore')
        ml = RE_LAB.search(h)
        chave = (ml.group(1) + '|' + ml.group(2).strip()) if ml else ''
    except urllib.error.HTTPError as e:
        chave = f'HTTP {e.code}'
    except Exception as e:
        chave = type(e).__name__
    if chave != '1|4':
        log(f'TESTE FALHOU ({chave or "página sem rótulo"}). Se a VPN está ligada, a BN está barrando '
            'o Python: avise o Claude para voltar à varredura pelo Chrome.')
        input('Enter para fechar.'); sys.exit(1)
    log('Teste OK (SC Cx. 1 Doc. 4 = página 37). Começando a fila.')
    for bib in (sys.argv[1:] or FILA):
        colecao(bib)
    log('Fila inteira concluída.')
