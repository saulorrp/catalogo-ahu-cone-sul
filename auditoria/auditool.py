import json
import tkinter as tk
from datetime import datetime
import os
import random

# CONFIGURAÇÕES
INPUT_FILE = "catalogo_ahu.json"
OUTPUT_FILE = "auditoria_class.json"
LIMIAR_SEGUNDOS = 3600  # 1 hora - se passar disso, considera nova sessão (delta=0)

class AuditApp:
    def __init__(self, root, data, existing_data):
        self.root = root
        self.data = data
        self.audited_data = existing_data
        self.index = 0
        self.total = len(data)
        
        self.root.title("Auditoria AHU - Qualis A1")
        self.root.configure(bg='black')
        self.root.geometry("900x700")

        # Contador
        self.counter_label = tk.Label(root, text="", fg="white", bg="black", font=("Consolas", 12))
        self.counter_label.pack(pady=10)

        # ID, Data e Reference Code
        self.info_label = tk.Label(root, text="", fg="#00FF00", bg="black", font=("Consolas", 11))
        self.info_label.pack(pady=5)

        # Descrição
        self.desc_label = tk.Label(root, text="", fg="white", bg="black", font=("Consolas", 14), wraplength=800, justify="left")
        self.desc_label.pack(pady=20)

        # Input
        self.var = tk.StringVar()
        self.var.trace_add("write", self.on_input_change)
        self.input_field = tk.Entry(root, font=("Consolas", 30), justify="center", textvariable=self.var)
        self.input_field.pack(pady=20)
        self.input_field.focus_set()

        self.load_item()

    def load_item(self):
        item = self.data[self.index]
        self.counter_label.config(text=f"{self.index + 1}/{self.total}")
        self.info_label.config(text=f"REF: {item.get('reference_code', 'N/A')} | ID: {item.get('document_id_and_date', 'N/A')}")
        self.desc_label.config(text=item.get('description', ''))
        self.input_field.delete(0, tk.END)

    def on_input_change(self, *args):
        val = self.var.get()
        if val in [str(i) for i in range(1, 10)]:
            self.save_and_next(int(val))

    def save_and_next(self, score):
        now = datetime.now()
        
        # Lógica de Tempo
        delta_seconds = 0
        if self.audited_data:
            last_entry = self.audited_data[-1]
            last_time = datetime.fromisoformat(last_entry['timestamp_raw'])
            delta = (now - last_time).total_seconds()
            # Se for menor que o limiar, salvamos o tempo. Se for maior, ignora (0)
            delta_seconds = delta if delta < LIMIAR_SEGUNDOS else 0
        
        item = self.data[self.index]
        
        audited_entry = {
            "description": item['description'],
            "reference_code": item['reference_code'],
            "vernacular_score": item['vernacular_score'],
            "human_score": score,
            "delta_seconds": round(delta_seconds, 2),
            "timestamp_raw": now.isoformat(),
            "timestamp_readable": now.strftime("%H:%M:%S, %d/%m/%Y")
        }
        
        self.audited_data.append(audited_entry)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.audited_data, f, indent=4, ensure_ascii=False)
        
        self.index += 1
        if self.index < self.total:
            self.load_item()
        else:
            self.root.destroy()
            print("Auditoria finalizada!")

def preparar_amostra():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    existing = []
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    
    # Filtra o que já foi processado para não repetir
    existing_refs = {item['reference_code'] for item in existing}
    subset = [d for d in all_data[10:] if d['reference_code'] not in existing_refs]
    
    n_amostra = int(len(all_data[10:]) * 0.1) + 10 - len(existing)
    
    random.seed(42)
    amostra = random.sample(subset, min(len(subset), n_amostra))
    
    return amostra, existing

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: {INPUT_FILE} não encontrado.")
    else:
        amostra_para_auditar, ja_auditados = preparar_amostra()
        if not amostra_para_auditar:
            print("Tudo auditado!")
        else:
            root = tk.Tk()
            app = AuditApp(root, amostra_para_auditar, ja_auditados)
            root.mainloop()