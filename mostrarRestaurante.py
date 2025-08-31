import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
import pickle
with open('restaurante.pkl', 'rb') as f:
  X_treino, X_teste, y_treino, y_teste = pickle.load(f)
  modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(accuracy_score(y_teste,previsoes))
from yellowbrick.classifier import ConfusionMatrix
confusion_matrix(y_teste, previsoes)
cm = ConfusionMatrix(modelo)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)
print(classification_report(y_teste, previsoes))
from sklearn import tree
previsores = X_treino.columns
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(23,7))
tree.plot_tree(modelo, feature_names=previsores, class_names = modelo.classes_, filled=True);
plt.show()