import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# ===============================
# 1. LOAD & FILTER DATA ANGKA
# ===============================
def load_emnist_digit(csv_path):
    print(f"Membaca {csv_path} ...")
    df = pd.read_csv(csv_path, header=None)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values

    # Ambil hanya label 0–9
    mask = y < 10
    X = X[mask]
    y = y[mask]

    # Normalisasi
    X = X.astype("float32") / 255.0

    images = []
    for i in range(len(X)):
        img = X[i].reshape(28, 28)
        img = np.fliplr(img)
        img = np.rot90(img)
        images.append(img)

    X = np.array(images).reshape(-1, 28, 28, 1)
    return X, y


# ===============================
# 2. LOAD DATA
# ===============================
X_train, y_train = load_emnist_digit("emnist-balanced-train.csv")
X_test, y_test = load_emnist_digit("emnist-balanced-test.csv")

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)


# ===============================
# 3. VISUALISASI DATA
# ===============================
plt.figure(figsize=(8, 2))
for i in range(5):
    idx = np.random.randint(0, len(X_train))
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[idx].reshape(28, 28), cmap="gray")
    plt.title(str(y_train[idx]))
    plt.axis("off")
plt.show()


# ===============================
# 4. MODEL CNN (DIGIT)
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")  # 10 kelas digit
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ===============================
# 5. TRAINING
# ===============================
print("Training CNN Digit...")
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test),
    verbose=1
)


# ===============================
# 6. EVALUASI & SIMPAN MODEL
# ===============================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Akurasi Test: {acc * 100:.2f}%")

model.save("model_ANGKA_cnn.h5")
print("Model berhasil disimpan sebagai model_ANGKA_cnn.h5")
