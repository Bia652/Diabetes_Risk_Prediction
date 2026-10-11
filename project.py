import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Carregar o Dataset (substitui 'diabetes_data.csv' pelo caminho real do teu ficheiro)
df = pd.read_csv('diabetes_risk.csv')

# 2. Configurações base 
sns.set_theme(style="whitegrid")
# Filtrar apenas as colunas numéricas e remover o ID (não é útil nas correlações/distribuições)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
if 'patient_id' in numeric_cols:
    numeric_cols.remove('patient_id')


# ==============================================================================
# Gráfico 1: Pairplot
# ==============================================================================
# Usa corner=True para evitar duplicação no espelho da matriz e facilitar leitura
plt.figure(figsize=(15, 15))
sns.pairplot(df[numeric_cols + ['diabetes_risk']], 
             hue='diabetes_risk', 
             palette='Set1', 
             corner=True)
plt.suptitle('Pairplot das Distribuições por Classe (Diabetes Risk)', y=1.02)
plt.show()


# ==============================================================================
# Gráfico 2: Subplots com Boxplots de cada Atributo por Classe
# ==============================================================================
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12)) # Ajusta as cols/rows ao nº de features
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    if i < len(axes):
        sns.boxplot(x='diabetes_risk', y=col, data=df, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Boxplot: {col}')
        axes[i].set_xlabel('')
        
# Ocultar eixos vazios (caso o número de features numéricas seja inferior a 9)
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# ==============================================================================
# Gráfico 3: Mapa de Correlação (Heatmap)
# ==============================================================================
plt.figure(figsize=(10, 8))
corr_matrix = df[numeric_cols].corr()

sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f', 
            vmin=-1, vmax=1, 
            linewidths=0.5)

plt.title('Mapa de Correlação entre Variáveis Numéricas')
plt.show()