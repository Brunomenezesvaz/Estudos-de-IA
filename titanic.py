import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# Carregar dados
base = pd.read_csv('train.csv', sep=',')

# Remover colunas não numéricas ou pouco úteis
base = base.drop(columns=["Name", "Ticket", "Cabin"])
print(base.columns)

# Verificar distribuição da classe alvo
np.unique(base["Survived"].astype(str), return_counts=True)
sns.countplot(x=base["Survived"]);

# Colunas categóricas para OneHot
cols_onehot_encode = ['Sex', 'Embarked']

# Inicializar OneHotEncoder
onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

# Aplicar o OneHotEncoder apenas nas colunas categóricas
df_onehot = onehot.fit_transform(base[cols_onehot_encode])

# Obter os novos nomes das colunas após a codificação
nomes_das_colunas = onehot.get_feature_names_out(cols_onehot_encode)

# Criar DataFrame com as variáveis codificadas
df_onehot = pd.DataFrame(df_onehot, columns=nomes_das_colunas, index=base.index)

# Juntar dados transformados com os numéricos originais
base_encoded = pd.concat([df_onehot, base.drop(columns=cols_onehot_encode)], axis=1)

# Definir X e y corretamente
X_prev = base_encoded.drop(columns=["Survived"])  # atributos
y_classe = base_encoded["Survived"]               # target

# Dividir treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X_prev, y_classe, test_size=0.20, random_state=42, stratify=y_classe
)

# Salvar no pickle
with open('titanic.pkl', mode='wb') as f:
    pickle.dump([X_treino, X_teste, y_treino, y_teste], f)