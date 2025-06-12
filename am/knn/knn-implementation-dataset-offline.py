#Importação das bibliotecas
#Numpy: utilizada para manipulação de arrays
import numpy as np
import pandas as pd
#Collections: utilizada para calcular a frequência dos k vizinhos
from collections import Counter
#sklearn: utilizada para leitura do dataset e divisão da base de dados
import sklearn
from sklearn.metrics import accuracy_score

#carregamento da base de dados iris
df = pd.read_csv('knn/dataset.csv')
#separação da base de dados (atributos = iris.data e classes = iris.target)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
#divisão da base de dados em treinamento e teste utilizando a proporção 80% treinamento e 20% teste (test_size=0.2 significa 20%)
#para congelar a aleatoridade, e o algoritmo usar sempre os mesmos dados para treinamento e teste use 'random_state=42' em train_test_split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.2)
#Escolha da quantidade de k vizinhos
k = 3 
#criação do array que irá armazenar as classes preditas
predictions = []
#permite analisar cada exemplo do teste
for x in X_test: 
    #cria o vetor com os valores do cálculo da distância
    distances = [] 
    #para cada exemplo da base de treinamento
    for x_train in X_train: 
        #calcula a distancia entre o exemplo de treinamento com o exemplo de teste por meio da medida da distância euclidiana
        distance = np.sqrt(np.sum((x - x_train) ** 2))
        #adiciona no vetor distances o valor da distancia calculado 
        distances.append(distance) 
#transforma o vetor de distances para o formato da biblioteca numpy
    distances = np.array(distances) 
    #guarda os indices do k vizinhos mais próximos após ordenar as distâncias do menor valor para o maior
    k_indices = np.argsort(distances)[:k] 
    #seleciona as classes dos vizinhos mais próximos
    k_nearest_labels = [y_train[i] for i in k_indices] 
    #realiza o processo de votação em que é escolhida a classe que mais aparece entre os vizinhos mais próximos
    most_common = Counter(k_nearest_labels).most_common(1) 
    #a classe mais votada é escolhida como classe predita
    predictions.append(most_common[0][0]) 
#transforma o vetor no formato da biblioteca numpy
predictions = np.array(predictions) 
#testa a acurácia do algoritmo
acc = accuracy_score(y_test, predictions)
#imprime as predições realizadas
print("Predições:", predictions)
print(f"Acurácia da classificação: {acc:.4f}")