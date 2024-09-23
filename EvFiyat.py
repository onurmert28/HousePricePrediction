# Gerekli kütüphaneleri yükle
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# California Housing veri setini yükle
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)  # Özellikler
y = pd.Series(housing.target)  # Hedef fiyatlar

# Veriyi eğitim ve test setlerine ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lineer Regresyon modelini oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Test seti üzerinde tahminler yap
y_pred = model.predict(X_test)

# Model performansını değerlendir
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sonuçları yazdır
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Gerçek fiyatlar ve tahmin edilen fiyatları karşılaştıran bir grafik oluştur
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color="blue", label="Tahmin edilen fiyatlar")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", lw=2, label="Mükemmel tahmin")
plt.xlabel("Gerçek Fiyatlar")
plt.ylabel("Tahmin Edilen Fiyatlar")
plt.title("Gerçek Fiyatlar vs Tahmin Edilen Fiyatlar")
plt.legend()
plt.show()
