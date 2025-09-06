import numpy as np
import matplotlib.pyplot as plt

# Generar datos de ejemplo
x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(scale=0.1, size=len(x))

plt.figure()
plt.scatter(x, y, alpha=0.7, label="Datos ruidosos")
plt.plot(x, np.sin(x), color='red', label='Funcion verdadera')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ejemplo de visualizacion de datos')
plt.legend()
plt.show()
