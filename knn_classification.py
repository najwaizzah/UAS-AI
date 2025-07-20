from sklearn.datasets import load_iris  # Load dataset bunga Iris
from sklearn.model_selection import train_test_split  # Untuk membagi data
from sklearn.neighbors import KNeighborsClassifier  # Algoritma k-NN
from sklearn.metrics import classification_report  # Laporan hasil klasifikasi

data = load_iris()  # Ambil dataset Iris
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)  
# Bagi data jadi 75% latih dan 25% uji

model = KNeighborsClassifier(n_neighbors=3)  # Inisialisasi model dengan 3 tetangga terdekat
model.fit(X_train, y_train)  # Latih model pada data latih

y_pred = model.predict(X_test)  # Prediksi kelas untuk data uji
print(classification_report(y_test, y_pred))  # Tampilkan metrik evaluasi (precision, recall, F1)
