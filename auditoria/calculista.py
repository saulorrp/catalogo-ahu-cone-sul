import json
import pandas as pd
import numpy as np

# CONFIGURAÇÕES
AUDIT_FILE = "auditoria_class.json"
CORPUS_FILE = "catalogo_ahu.json"

def analisar_resultados():
    # 1. Carregar Dados
    with open(AUDIT_FILE, 'r', encoding='utf-8') as f:
        audit_data = json.load(f)
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
    
    df_audit = pd.DataFrame(audit_data)
    df_corpus = pd.DataFrame(corpus_data)
    
    # Garantir que são numéricos
    df_audit['vernacular_score'] = pd.to_numeric(df_audit['vernacular_score'])
    df_audit['human_score'] = pd.to_numeric(df_audit['human_score'])
    df_audit['delta_seconds'] = pd.to_numeric(df_audit['delta_seconds'])

    # --- (a) Comparação IA vs Humano ---
    mae = np.mean(np.abs(df_audit['vernacular_score'] - df_audit['human_score']))
    correlacao = df_audit['vernacular_score'].corr(df_audit['human_score'])
    # Viés médio: positivo significa que a IA tende a dar notas maiores que o humano
    bias_medio = np.mean(df_audit['vernacular_score'] - df_audit['human_score'])

    # --- (b) Leitura Média de Tempo ---
    tempo_medio = df_audit['delta_seconds'].mean()
    tempo_total = df_audit['delta_seconds'].sum() / 3600 # em horas

    # --- (c) & (d) Estatísticas do Corpus Geral ---
    distribuicao = df_corpus['vernacular_score'].value_counts().sort_index()
    porcentagem = df_corpus['vernacular_score'].value_counts(normalize=True).sort_index() * 100

    # Exibição dos Resultados
    print("--- RELATÓRIO DE AUDITORIA E CONSISTÊNCIA ---\n")
    
    print(f"1. AUDITORIA DE PRECISÃO (IA vs HUMANO):")
    print(f"   - MAE (Erro Médio Absoluto): {mae:.2f} (em uma escala de 1-9)")
    print(f"   - Correlação de Pearson: {correlacao:.2f}")
    print(f"   - Viés Sistemático: {bias_medio:.2f} (IA tende a {'superestimar' if bias_medio > 0 else 'subestimar'})\n")

    print(f"2. EFICIÊNCIA DE AUDITORIA:")
    print(f"   - Tempo médio de decisão humana: {tempo_medio:.2f} segundos/doc")
    print(f"   - Tempo total dedicado à auditoria: {tempo_total:.2f} horas\n")

    print(f"3. DISTRIBUIÇÃO DO CORPUS (N={len(df_corpus)}):")
    print("Score | Contagem | Porcentagem")
    for score in range(1, 10):
        count = distribuicao.get(score, 0)
        perc = porcentagem.get(score, 0)
        print(f"  {score}   |   {count:5}  |    {perc:.2f}%")

    # --- (e) Informações Inteligentes Extras ---
    print("\n4. ANÁLISE COMPORTAMENTAL (CRIATIVA):")
    # Qual score a IA mais errou (maior dispersão)?
    df_audit['erro'] = abs(df_audit['vernacular_score'] - df_audit['human_score'])
    erro_por_score = df_audit.groupby('human_score')['erro'].mean()
    print(f"   - Score humano com maior incerteza (IA erra mais): Nota {erro_por_score.idxmax()} (Média de erro: {erro_por_score.max():.2f})")
    
    # Salvar relatório em texto
    with open("relatorio_auditoria_final.txt", "w", encoding="utf-8") as f:
        f.write("RELATÓRIO DE VALIDAÇÃO DE MODELO\n")
        f.write(f"Concordância IA-Humano (MAE): {mae:.2f}\n")
        f.write(f"Tempo médio por documento: {tempo_medio:.2f}s\n")
        f.write(f"Distribuição Score: {distribuicao.to_dict()}\n")

if __name__ == "__main__":
    analisar_resultados()