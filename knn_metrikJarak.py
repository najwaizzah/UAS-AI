import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data = load_iris()
X, y = data.data, data.target
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
query = X_scaled[0]

cos_dist = cosine_distances([query], X_scaled)[0]
cos_indices = np.argsort(cos_dist)[1:4]
print("Top-3 Cosine:", cos_indices)

def correlation_distance(a, b):
    return 1 - pearsonr(a, b)[0]

corr_dist = [correlation_distance(query, x) for x in X_scaled]
corr_indices = np.argsort(corr_dist)[1:4]
print("Top-3 Correlation:", corr_indices)
