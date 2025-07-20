from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from time import time

data = load_iris()
X, y = data.data, data.target
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

algorithms = ['brute', 'kd_tree', 'ball_tree']
for algo in algorithms:
    start = time()
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"{algo} - Accuracy: {acc:.4f} | Time: {time() - start:.4f} s")
