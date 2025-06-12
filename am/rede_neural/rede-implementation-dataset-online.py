import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def inicializa_pesos(n_inputs):
    # pesos[0] é o bias; pesos[1:] são os pesos das entradas
    return np.zeros(n_inputs + 1)

def ativacao(soma):
    # passo de ativação: retorna 1 se soma >= 0, senão 0
    return 1 if soma >= 0 else 0

def atualiza_pesos(pesos, x_i, y_i, y_pred, lr):
    erro = y_i - y_pred
    pesos[1:] += lr * erro * x_i
    pesos[0]   += lr * erro  # bias
    return pesos

def treina_perceptron(X, y, lr=0.1, n_epochs=10):
    pesos = inicializa_pesos(X.shape[1])
    for epoca in range(n_epochs):
        for x_i, y_i in zip(X, y):
            soma = np.dot(pesos[1:], x_i) + pesos[0]
            y_pred = ativacao(soma)
            pesos = atualiza_pesos(pesos, x_i, y_i, y_pred, lr)
    return pesos

def prediz_perceptron(X, pesos):
    resultados = []
    for x_i in X:
        soma = np.dot(pesos[1:], x_i) + pesos[0]
        resultados.append(ativacao(soma))
    return np.array(resultados)

#Dados de entrada do algoritmo
# Carrega o dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Filtra apenas as classes 0 e 1 (para binário)
X = X[y != 2]
y = y[y != 2]

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Treinamento
pesos_finais = treina_perceptron(X_train, y_train)
print("Pesos finais:", pesos_finais)

# Predição
preds = prediz_perceptron(X_test, pesos_finais)
print("Previsões:", preds)
print("Acurácia:", np.mean(preds == y_test))