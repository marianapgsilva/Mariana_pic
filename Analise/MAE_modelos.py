"""
Seleção de Variáveis (Feature Selection)
================================================
Este script testa diferentes "Cenários" de variáveis clínicas em modelos 
baseline não-recorrentes (Random Forest e MLP) para isolar o ruído e 
descobrir quais as comorbilidades com verdadeiro valor preditivo.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. PREPARAÇÃO E LIMPEZA (Foco em comparação de variáveis)
# ============================================================
def preparar_dados(caminho):
    if not os.path.exists(caminho): return None
    df = pd.read_csv(caminho)
    
    df['genero_num'] = df['Género'].astype(str).str.strip().str[0].str.upper().map({'F': 0, 'M': 1})
    df['idade'] = pd.to_numeric(df['Idade_anos (à data cirurgia)'], errors='coerce')
    for c in ['Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    
    df['imc_0'] = pd.to_numeric(df['IMC_kg/m2'], errors='coerce')
    df['imc_3m'] = pd.to_numeric(df['IMC_3_mes'], errors='coerce')
    df['imc_12m'] = pd.to_numeric(df['IMC_12_mes'], errors='coerce')

    cols_total = ['genero_num', 'idade', 'imc_0', 'imc_3m', 'Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC', 'imc_12m']
    return df.dropna(subset=cols_total).copy()

# ============================================================
# 2. TREINO E AVALIAÇÃO (Modelos Baseline Apenas)
# ============================================================
def comparar_cenarios(df):
    # Definição dos Cenários
    features_A = ['genero_num', 'idade', 'imc_0', 'imc_3m', 'Diabetes', 'HTA', 'Dislipidemia', 'SAOS_DPOC']
    features_B = ['genero_num', 'idade', 'imc_0', 'imc_3m']
    features_C = ['genero_num', 'idade', 'imc_0', 'imc_3m', 'Dislipidemia'] 
    target = 'imc_12m'

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    resultados = []

    def avaliar(X_cols, nome_cenario):
        X_train, y_train = df_train[X_cols], df_train[target]
        X_test, y_test = df_test[X_cols], df_test[target]

        # Normalização
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # 1. Random Forest (Padrão-ouro para Feature Selection)
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
        mae_rf = mean_absolute_error(y_test, rf.predict(X_test))

        # 2. MLP (Validador Neural Estático)
        mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42).fit(X_train_sc, y_train)
        mae_mlp = mean_absolute_error(y_test, mlp.predict(X_test_sc))

        return {"Cenário": nome_cenario, "RF": mae_rf, "MLP": mae_mlp}

    resultados.append(avaliar(features_A, "A (Com Tudo)"))
    resultados.append(avaliar(features_B, "B (Sem Comorb)"))
    resultados.append(avaliar(features_C, "C (Só Dislipid)"))

    # Mostrar Tabela
    res_df = pd.DataFrame(resultados).set_index("Cenário")
    print("\n=== COMPARAÇÃO DE ERRO (MAE em kg/m²) ===")
    print(res_df.round(4))
    
    # Cálculo da Diferença B - A
    diff = res_df.loc["B (Sem Comorb)"] - res_df.loc["A (Com Tudo)"]
    print("\n=== PENALIZAÇÃO PELO DESCARTE (B - A) ===")
    print(diff.round(4))
    
    # Cálculo da Diferença C - A
    diff = res_df.loc["C (Só Dislipid)"] - res_df.loc["A (Com Tudo)"]
    print("\n=== PENALIZAÇÃO PELO DESCARTE (C - A) ===")
    print(diff.round(4))
    
# ============================================================
# EXECUÇÃO
# ============================================================
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJETO_ROOT = os.path.join(BASE_DIR, "..")
    CAMINHO_FINAL = os.path.join(PROJETO_ROOT, "Dados", "Base_dados_v1.csv")

    print(f"--- Iniciando Análise de Variáveis (Feature Selection) ---")
    
    dados = preparar_dados(CAMINHO_FINAL)
    
    if dados is not None:
        print(f"✓ {len(dados)} doentes prontos para comparação.")
        comparar_cenarios(dados)
    else:
        print(f"❌ Erro: Não foi possível carregar os dados em {CAMINHO_FINAL}")