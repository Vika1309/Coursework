import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

print("Завантаження датасету MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("Створення моделі...")
model = models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])


model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

print("Початок навчання...")
history = model.fit(x_train, y_train_cat, 
                    batch_size=128, 
                    epochs=10, 
                    validation_split=0.2) 


model.save("mnist_cnn.h5")
print("Модель успішно збережена як mnist_cnn.h5")


test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Точність на тестовій вибірці: {test_acc:.4f}")

y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred))

os.makedirs("plots", exist_ok=True)

# Графік Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Графік функції втрат (Loss)')
plt.xlabel('Епоха')
plt.ylabel('Loss')
plt.legend()
plt.savefig("plots/loss_plot.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Графік точності (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("plots/accuracy_plot.png")
plt.close()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матриця помилок (Confusion Matrix)')
plt.xlabel('Передбачений клас')
plt.ylabel('Справжній клас')
plt.savefig("plots/confusion_matrix.png")
plt.close()
