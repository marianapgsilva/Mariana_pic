"""
Análise SHAP - Importância de Variáveis (Feature Importance)
================================================
Este script identifica quais as variáveis clínicas que têm maior impacto 
matemático na previsão da trajetória de peso.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJETO_ROOT = os.path.join(BASE_DIR, "..")

try:
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError as e:
    print(f"\n[ERRO] Falta instalar bibliotecas no venv: {e}")
    import sys
    sys.exit()

# ============================================================
# 1. PREPARAÇÃO E LIMPEZA
# ============================================================
def preparar_dados(caminho):
    if not os.path.exists(caminho):
        print(f"\n[ERRO] Ficheiro não encontrado em: {caminho}")
        return None

    df = pd.read_csv(caminho)

    df['genero_num'] = df['Género'].astype(str).str.strip().str[0].str.upper().map({'F': 0, 'M': 1})
    df['idade'] = pd.to_numeric(df['Idade_anos (à data cirurgia)'], errors='coerce')
    
    comorbs = ['Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC']
    for c in comorbs:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)

    df['imc_0'] = pd.to_numeric(df['IMC_kg/m2'], errors='coerce')
    df['imc_3m'] = pd.to_numeric(df['IMC_3_mes'], errors='coerce')
    df['imc_12m'] = pd.to_numeric(df['IMC_12_mes'], errors='coerce')

    cols_modelo = ['genero_num', 'idade', 'imc_0', 'imc_3m', 'Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC', 'imc_12m']
    
    df_limpo = df.dropna(subset=cols_modelo).copy()
    df_limpo = df_limpo.replace([np.inf, -np.inf], np.nan).dropna(subset=cols_modelo)
    
    print(f"✓ Doentes originais: {len(df)}")
    print(f"✓ Doentes após limpeza de nulos: {len(df_limpo)}")
        
    return df_limpo

# ============================================================
# 2. ANÁLISE SHAP (Abertura da Caixa Preta)
# ============================================================
def analisar_modelos(df):
    features = ['genero_num', 'idade', 'imc_0', 'imc_3m', 'Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC']
    X = df[features]
    y = df['imc_12m']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nA treinar modelos baseline...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42).fit(X_scaled, y)

    print("\nA gerar gráficos SHAP de Importância...")
    
    # --- SHAP Random Forest ---
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf = explainer_rf.shap_values(X)
    
    plt.figure(figsize=(10, 5))
    plt.title("Importância de Variáveis: Random Forest", fontsize=14, fontweight='bold')
    shap.summary_plot(shap_values_rf, X, plot_type="bar", show=False)
    plt.tight_layout()
    caminho_rf = os.path.join(BASE_DIR, "shap_importancia_RF.png")
    plt.savefig(caminho_rf, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico RF salvo em: {caminho_rf}")
    plt.show()

    # --- SHAP MLP (Rede Neuronal) ---
    print("\nA processar SHAP para a Rede Neuronal (pode demorar uns segundos)...")
    amostra = shap.sample(X_scaled, 50) # Amostra para acelerar o KernelExplainer
    expl_mlp = shap.KernelExplainer(mlp.predict, amostra)
    s_mlp = expl_mlp.shap_values(X_scaled)
    
    plt.figure(figsize=(10, 5))
    plt.title("Importância de Variáveis: Rede Neuronal (MLP)", fontsize=14, fontweight='bold')
    # O SHAP do Kernel explainer precisa dos nomes originais
    shap.summary_plot(s_mlp, features=X, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    caminho_mlp = os.path.join(BASE_DIR, "shap_importancia_MLP.png")
    plt.savefig(caminho_mlp, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico MLP salvo em: {caminho_mlp}")
    plt.show()

# ============================================================
# EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    CAMINHO_FINAL = os.path.join(PROJETO_ROOT, "Dados", "Base_dados_v1.csv") 

    print(f"--- Iniciando Análise SHAP ---")
    dados = preparar_dados(CAMINHO_FINAL)
    
    if dados is not None:
        analisar_modelos(dados)