import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# =========================
# PATH MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn_ocr_emnist_csv.h5")

model = load_model(MODEL_PATH)

# =========================
# LABEL MAP EMNIST BALANCED
# =========================
label_map = {
    0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
    10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',16:'G',17:'H',18:'I',19:'J',
    20:'K',21:'L',22:'M',23:'N',24:'O',25:'P',26:'Q',27:'R',28:'S',29:'T',
    30:'U',31:'V',32:'W',33:'X',34:'Y',35:'Z',
    36:'a',37:'b',38:'d',39:'e',40:'f',41:'g',42:'h',43:'n',44:'q',45:'r',46:'t'
}

# =========================
# PREPROCESS SESUAI TRAINING
# =========================
def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan!")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, th = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("Karakter tidak terdeteksi")

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    roi = th[y:y+h, x:x+w]

    pad = max(w, h) // 4
    roi = cv2.copyMakeBorder(
        roi, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=0
    )

    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 28, 28, 1)

    return img, roi

# =========================
# VISUAL OUTPUT
# =========================
def visualize(original, processed, probs):
    top5 = np.argsort(probs)[-5:][::-1]
    labels = [label_map[i] for i in top5]
    values = probs[top5] * 100

    pred_char = labels[0]
    conf = values[0]

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(processed.reshape(28,28), cmap="gray")
    plt.title("Gambar Diproses")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.axis("off")
    plt.text(0.1, 0.6, f"Karakter: {pred_char}",
             fontsize=22, color="green")
    plt.text(0.1, 0.4, f"Confidence: {conf:.2f}%",
             fontsize=14)

    plt.subplot(2, 1, 2)
    plt.barh(labels[::-1], values[::-1], color="green")
    plt.xlabel("Confidence (%)")
    plt.title("Top 5 Prediksi")
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()

# =========================
# MAIN
# =========================
def predict_image(image_path):
    original, processed = preprocess_image(image_path)
    probs = model.predict(processed, verbose=0)[0]
    visualize(original, processed, probs)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    path = r"D:\PCD2\test image\B kecil.png"
    predict_image(path)
