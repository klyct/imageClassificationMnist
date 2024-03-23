import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# cargamos conjunto de datos
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizamos los datos
x_train, x_test = x_train / 255.0, x_test / 255.0

# construccion del modelo
# dedinimoa la arquitectura del modelo de la red neuronal (cnn)
model = models.Sequential([
    # es la primera capa convolucional, tiene 32 filtros y es de 3x3
    # espera imagenes de 28x28 en escala de grises
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    # aplicamos una capa de aprupamiento maximum para reducir dimensionalidad
    # manteniendo las caracteristicas imporantes
    layers.MaxPooling2D((2, 2)),
    # una sefunda capa convolucinal
    layers.Conv2D(64, (3, 3), activation='relu'),
    # volvemos a agrupar
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    # esta acapa aolana la salida para alimentar una capa densa
    layers.Flatten(),
    # es una capa densa se 64 neuronas
    layers.Dense(64, activation='relu'),
    # capa de salida con 10 neuronas (una para cada digito)
    # softman es para clasificacion multiclase
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(x_train.reshape(-1, 28, 28, 1), y_train, epochs=5, batch_size=64)

# Evaluacion del modelo
test_loss, test_acc = model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test)
print(f'Accuracy: {test_acc}')
