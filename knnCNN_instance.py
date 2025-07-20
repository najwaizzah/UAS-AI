from imblearn.under_sampling import CondensedNearestNeighbour
from sklearn.neighbors import KNeighborsClassifier

cnn = CondensedNearestNeighbour()
X_resampled, y_resampled = cnn.fit_resample(X_train, y_train)
print("Sebelum:", len(X_train), "Sesudah:", len(X_resampled))

model1 = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
model2 = KNeighborsClassifier(n_neighbors=3).fit(X_resampled, y_resampled)

print("Akurasi awal:", model1.score(X_test, y_test))
print("Akurasi CNN:", model2.score(X_test, y_test))
