import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report

# 1. Завантаження та підготовка даних
print("Завантаження датасету MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормалізація та зміна форми для CNN (додаємо канал кольору)
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Кодування категоріальних ознак (One-Hot Encoding)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Архітектура нейронної мережі (CNN)
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

# Компіляція моделі
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# 3. Навчання моделі
print("Початок навчання...")
history = model.fit(x_train, y_train_cat, 
                    batch_size=128, 
                    epochs=10, 
                    validation_split=0.2) # Використовуємо 20% тренувальних даних для валідації

# Збереження навченої моделі для застосунку
model.save("mnist_cnn.h5")
print("Модель успішно збережена як mnist_cnn.h5")

# 4. Оцінка моделі на тестовій вибірці
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)
print(f"Точність на тестовій вибірці: {test_acc:.4f}")

# Передбачення для побудови метрик
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Генерація звіту з метриками (Precision, Recall, F1-score)
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred))

# 5. Побудова та збереження графіків (для звіту)
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

# Графік Accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Графік точності (Accuracy)')
plt.xlabel('Епоха')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("plots/accuracy_plot.png")
plt.close()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Матриця помилок (Confusion Matrix)')
plt.xlabel('Передбачений клас')
plt.ylabel('Справжній клас')
plt.savefig("plots/confusion_matrix.png")
plt.close()

print("Графіки збережено у папку 'plots'.")