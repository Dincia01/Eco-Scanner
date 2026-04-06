import matplotlib.pyplot as plt

# Aquí colocamos los datos que te soltó la terminal (los copié de tu mensaje)
epochs = list(range(1, 11))
loss_values = [1.0030, 0.8004, 0.7488, 0.7460, 0.7336, 0.7075, 0.7009, 0.7050, 0.7265, 0.7108]
acc_values = [79.04, 78.31, 80.51, 80.07, 80.51, 81.69, 81.03, 81.25, 81.84, 82.06]

# Crear la figura con dos gráficos
plt.figure(figsize=(12, 5))

# 1. Gráfico de Pérdida (Loss) - Debe ir hacia abajo
plt.subplot(1, 2, 1)
plt.plot(epochs, loss_values, 'r-o', label='Pérdida (Loss)')
plt.title('Progreso del Error (Loss)')
plt.xlabel('Épocas')
plt.ylabel('Valor de Pérdida')
plt.grid(True)
plt.legend()

# 2. Gráfico de Precisión (Accuracy) - Debe ir hacia arriba
plt.subplot(1, 2, 2)
plt.plot(epochs, acc_values, 'b-s', label='Precisión (Acc)')
plt.title('Progreso de la Precisión (%)')
plt.xlabel('Épocas')
plt.ylabel('Porcentaje %')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('resultado_entrenamiento.png') # Guarda la imagen automáticamente
plt.show()