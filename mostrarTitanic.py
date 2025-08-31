import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import pickle

# Carregar dados pré-processados
with open('titanic.pkl', 'rb') as f:
    X_treino, X_teste, y_treino, y_teste = pickle.load(f)

# Criar o modelo de árvore de decisão
modelo = DecisionTreeClassifier(criterion='entropy')
modelo.fit(X_treino, y_treino)

# Fazer previsões
previsoes = modelo.predict(X_teste)

# Avaliar desempenho
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Acurácia:", accuracy_score(y_teste, previsoes))
print(classification_report(y_teste, previsoes))

# Matriz de confusão com Yellowbrick
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(modelo, classes=[str(c) for c in modelo.classes_])
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

# Plotar árvore de decisão
from sklearn import tree
previsores = X_treino.columns
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(23, 7))
tree.plot_tree(
    modelo,
    feature_names=previsores,
    class_names=[str(c) for c in modelo.classes_],  # ⚡ converte para string
    filled=True
)
plt.show()