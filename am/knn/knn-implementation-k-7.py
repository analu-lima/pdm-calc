import numpy as np
from collections import Counter
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

cancer = fetch_ucirepo(id=17)
X = cancer.data.features[['radius1', 'concavity1']].values
y = cancer.data.targets.values.ravel()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

k = 7
predictions = []

for x in X_test:
    distances = []
    for x_train in X_train:
        distance = np.sqrt(np.sum((x - x_train) ** 2))
        distances.append(distance)

    distances = np.array(distances)
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    predictions.append(most_common[0][0])

predictions = np.array(predictions)

acc = accuracy_score(y_test, predictions)
print("✅ Predições:", predictions)
print(f"🎯 Acurácia da classificação: {acc:.4f}")