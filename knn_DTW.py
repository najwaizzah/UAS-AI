# Import pustaka yang dibutuhkan
from tslearn.datasets import CachedDatasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from sklearn.metrics import classification_report

# 1. Load dataset time-series dari tslearn
X_train, y_train, X_test, y_test = CachedDatasets().load_dataset("Trace")

# 2. Inisialisasi model k-NN dengan 1 tetangga dan metrik DTW
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")

# 3. Training model
clf.fit(X_train, y_train)

# 4. Prediksi pada data uji
y_pred = clf.predict(X_test)

# 5. Evaluasi hasil prediksi
print("=== Classification Report (DTW) ===")
print(classification_report(y_test, y_pred))

# 6. Akurasi secara keseluruhan
accuracy = clf.score(X_test, y_test)
print(f"DTW Accuracy: {accuracy:.4f}")
