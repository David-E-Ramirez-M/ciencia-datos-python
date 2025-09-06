import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Datos de ejemplo
np.random.seed(42)
data = {
    "x": np.arange(10),
    "y": 3 * np.arange(10) + np.random.randn(10)
}

df = pd.DataFrame(data)

# Estadísticas descriptivas
print("Estadisticas descriptivas:")
print(df.describe())

# Ajustar un modelo de regresión lineal
X = df[["x"]]
y = df["y"]
model = LinearRegression()
model.fit(X, y)

print("Pendiente:", model.coef_[0])
print("Intercepto:", model.intercept_)
